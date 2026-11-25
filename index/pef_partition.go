// Copyright 2026 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package index

import (
	"fmt"
	"math/bits"
)

type pefChunkKind uint8

const (
	pefKindAllOnes pefChunkKind = 0
	pefKindEF      pefChunkKind = 1
	pefKindBitmap  pefChunkKind = 2
)

// One discriminator bit selects between EF and bitmap; all-ones is
// implicit (chosen iff pefAllOnesBitsize returns 0).
const pefCodecTypeBits = 1

// pefEFBitsizePaper returns the EF cost of encoding n keys into a chunk
// whose relative key range is [0, lastRel] (inclusive), in bits,
// without sampling overhead:
//
//	bits = n*ℓ + (n + ⌊(lastRel+1) / 2^ℓ⌋ + 2),  ℓ = ⌊log₂((lastRel+1) / n)⌋.
func pefEFBitsizePaper(lastRel, n uint64) uint64 {
	if n == 0 {
		return 0
	}
	var lo uint64
	if lastRel >= n {
		lo = uint64(bits.Len64(lastRel/n) - 1)
	}
	higher := n + (lastRel >> lo) + 1 + 2
	return higher + n*lo
}

// pefBitmapBitsizePaper returns the cost of a flat bitmap chunk:
// universe = lastRel+1 bits.
func pefBitmapBitsizePaper(lastRel, _ uint64) uint64 {
	if lastRel == ^uint64(0) {
		return ^uint64(0)
	}
	return lastRel + 1
}

// pefAllOnesBitsize is 0 when the chunk is a contiguous run of n integers
// (i.e. lastRel == n-1), else infinity.
func pefAllOnesBitsize(lastRel, n uint64) uint64 {
	if n > 0 && lastRel == n-1 {
		return 0
	}
	return ^uint64(0)
}

// pefMinCodecBitsize picks the cheapest of {all-ones, EF, bitmap} for a
// chunk of n keys whose inclusive relative range is [0, lastRel].
func pefMinCodecBitsize(lastRel, n uint64) uint64 {
	if n > 0 && lastRel == n-1 {
		return 0
	}
	ef := pefEFBitsizePaper(lastRel, n) + pefCodecTypeBits
	bm := pefBitmapBitsizePaper(lastRel, n)
	if bm < ^uint64(0) && bm+pefCodecTypeBits < ef {
		return bm + pefCodecTypeBits
	}
	return ef
}

// pefSelectCodec picks the cheapest codec for a chunk whose inclusive
// relative range is [0, lastRel] and contains n keys.
func pefSelectCodec(lastRel, n uint32) pefChunkKind {
	lr, nn := uint64(lastRel), uint64(n)
	best := pefAllOnesBitsize(lr, nn)
	kind := pefKindAllOnes
	if ef := pefEFBitsizePaper(lr, nn) + pefCodecTypeBits; ef < best {
		best = ef
		kind = pefKindEF
	}
	bm := pefBitmapBitsizePaper(lr, nn)
	if bm < ^uint64(0) && bm+pefCodecTypeBits < best {
		kind = pefKindBitmap
	}
	return kind
}

const (
	defaultPEFFixCost = uint64(112)
	defaultPEFEps1    = 0.03
	defaultPEFEps2    = 0.3
	defaultPEFEps3    = 0.01
)

// pefPartitionConfig tunes the (1+ε)-approximate optimal-partition DP.
type pefPartitionConfig struct {
	Eps1, Eps2, Eps3 float64
	FixCost          uint64
}

func defaultPEFPartitionConfig() pefPartitionConfig {
	return pefPartitionConfig{
		Eps1:    defaultPEFEps1,
		Eps2:    defaultPEFEps2,
		Eps3:    defaultPEFEps3,
		FixCost: defaultPEFFixCost,
	}
}

func (cfg pefPartitionConfig) validate() error {
	if cfg.Eps1 < 0 || cfg.Eps1 >= 1 {
		return fmt.Errorf("pefPartitionConfig.Eps1 must be in [0, 1), got %v", cfg.Eps1)
	}
	if cfg.Eps2 <= 0 {
		return fmt.Errorf("pefPartitionConfig.Eps2 must be > 0, got %v", cfg.Eps2)
	}
	if cfg.Eps3 < 0 {
		return fmt.Errorf("pefPartitionConfig.Eps3 must be >= 0, got %v", cfg.Eps3)
	}
	if cfg.FixCost == 0 {
		return fmt.Errorf("pefPartitionConfig.FixCost must be > 0")
	}
	return nil
}

func pefSuperblockSize(fixCost uint64, eps3 float64) int {
	if eps3 <= 0 {
		return 1 << 30
	}
	sz := int(float64(fixCost) / eps3)
	if sz < 1 {
		return 1
	}
	return sz
}

type pefCostWindow struct {
	keys           []uint32
	start, end     uint32
	minP, maxP     uint64
	costUpperBound uint64
}

func (w *pefCostWindow) lastRel() uint64 { return w.maxP - w.minP }
func (w *pefCostWindow) size() uint32    { return w.end - w.start }

func (w *pefCostWindow) advanceStart() {
	w.minP = uint64(w.keys[w.start]) + 1
	w.start++
}

func (w *pefCostWindow) advanceEnd() {
	w.maxP = uint64(w.keys[w.end])
	w.end++
}

// pefPartitionScratch holds reusable buffers for sequential superblock DP calls.
type pefPartitionScratch struct {
	minCost []uint64
	path    []uint32
	windows []pefCostWindow
}

func (s *pefPartitionScratch) reset(size uint32) {
	need := int(size) + 1
	if cap(s.minCost) < need {
		s.minCost = make([]uint64, need)
	} else {
		s.minCost = s.minCost[:need]
	}
	if cap(s.path) < need {
		s.path = make([]uint32, need)
	} else {
		s.path = s.path[:need]
		clear(s.path)
	}
	s.windows = s.windows[:0]
}

// partition runs the (1+ε)-approximate optimal partition DP sequentially across
// superblocks on strictly increasing keys, returning ascending exclusive end
// indices in [1, len(keys)] appended to out[:0].
func (s *pefPartitionScratch) partition(keys []uint32, cfg pefPartitionConfig, out []uint32) []uint32 {
	if err := cfg.validate(); err != nil {
		panic(err)
	}
	out = out[:0]
	n := len(keys)
	if n == 0 {
		return out
	}
	if n == 1 {
		return append(out, 1)
	}

	superSize := pefSuperblockSize(cfg.FixCost, cfg.Eps3)
	superPos := 0
	superBase := uint64(keys[0])
	for superPos < n {
		sz := superSize
		if sz > n-superPos {
			sz = n - superPos
		}
		// Merge a tiny tail into the current superblock so no superblock is
		// shorter than superSize (mirrors PISA compute_partition).
		if rem := n - (superPos + sz); rem > 0 && rem < superSize {
			sz = n - superPos
		}
		superKeys := keys[superPos : superPos+sz]
		superLastKey := uint64(superKeys[sz-1])
		out = s.computeSuperblock(superKeys, superBase, superLastKey, cfg, uint32(superPos), out)
		superPos += sz
		superBase = superLastKey + 1
	}
	return out
}

func (s *pefPartitionScratch) computeSuperblock(
	keys []uint32,
	base, lastKey uint64,
	cfg pefPartitionConfig,
	offset uint32,
	out []uint32,
) []uint32 {
	size := uint32(len(keys))
	if size == 0 {
		return out
	}
	s.reset(size)

	singleBlockCost := pefMinCodecBitsize(lastKey-base, uint64(size)) + cfg.FixCost
	for i := range s.minCost {
		s.minCost[i] = singleBlockCost
	}
	s.minCost[0] = 0

	costLb := pefMinCodecBitsize(1, 1) + cfg.FixCost
	costBound := costLb
	for {
		if cfg.Eps1 > 0 && float64(costBound) >= float64(costLb)/cfg.Eps1 {
			break
		}
		s.windows = append(s.windows, pefCostWindow{
			keys: keys, minP: base, costUpperBound: costBound,
		})
		if costBound >= singleBlockCost {
			break
		}
		next := uint64(float64(costBound) * (1 + cfg.Eps2))
		if next == costBound {
			next++
		}
		costBound = next
	}

	for i := uint32(0); i < size; i++ {
		lastEnd := i + 1
		for w := range s.windows {
			win := &s.windows[w]
			for win.end < lastEnd {
				win.advanceEnd()
			}
			for {
				wc := pefMinCodecBitsize(win.lastRel(), uint64(win.size())) + cfg.FixCost
				if s.minCost[i]+wc < s.minCost[win.end] {
					s.minCost[win.end] = s.minCost[i] + wc
					s.path[win.end] = i
				}
				lastEnd = win.end
				if win.end == size || wc >= win.costUpperBound {
					break
				}
				win.advanceEnd()
			}
			win.advanceStart()
		}
	}

	startLen := len(out)
	for curr := size; curr != 0; curr = s.path[curr] {
		out = append(out, offset+curr)
	}
	for i, j := startLen, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out
}

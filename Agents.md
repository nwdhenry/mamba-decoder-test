# MambaGPT Agent Guidelines

## Repository Structure & Focus Areas
```
.
├── mamba_ssm/             **PRIMARY WORK DIRECTORY**
│   ├── models/            *Model implementations*
│   │   ├── mamba_gpt.py   **CORE FILE** (Architecture)
│   │   └── config_mamba.py
│   ├── training/          *Training scripts*
│   │   └── train_mambagpt.py **CORE FILE**
│   ├── ops/               *Custom CUDA kernels*
│   └── utils/             *Helper functions*
├── tests/                 *Validation tests*
├── benchmarks/            *Performance metrics*
├── csrc/                  *C++/CUDA source (advanced)*
└── evals/                 *Evaluation metrics*
```

## Contribution Guidelines
1. **VRAM Constraint First**: All changes must maintain <12GB VRAM usage
2. **Prefer CPU-Compatible Code**: Assume no CUDA access during development
3. **Modular Development**:
   - New features → Add behind feature flags
   - Optimizations → Isolate in `/ops` or `/utils`
4. **Testing Mandatory**: 100% test coverage for VRAM-critical components

## Development Workflow
### For Model Changes:
```markdown
1. Modify `/mamba_ssm/models/mamba_gpt.py`
2. Update configs in `/mamba_ssm/models/config_mamba.py`
3. Validate with CPU tests: `pytest tests/test_model_cpu.py`
```

### For Training Changes:
```markdown
1. Modify `/mamba_ssm/training/train_mambagpt.py`
2. Add VRAM monitoring hooks
3. Test with: `python -m mamba_ssm.training.train_mambagpt --cpu-mode`
```

## Validation Protocol
**Before Every Commit:**
```bash
# Run core tests (CPU only)
pytest tests/test_model_cpu.py tests/training/test_trainer_cpu.py

# Check memory constraints
python benchmarks/vram_simulator.py --model base

# Verify parameter count
python -m mamba_ssm.utils.param_counter --config base
```

**Performance Testing:**
```bash
# CPU latency benchmark
python benchmarks/latency.py --device cpu --context 2048

# Memory footprint test
python benchmarks/memory_profile.py --mode train --batch 4
```

## PR & Commit Standards
**PR Title Format:**  
`[Scope] Brief description`  
Examples:  
`[MODEL] Add state caching for inference`  
`[TRAIN] Implement 8-bit Adam optimizer`

**PR Body Must Include:**
1. VRAM impact analysis
2. CPU validation results
3. Affected components diagram
4. Risk assessment for OOM scenarios

## Agent-Specific Directives
1. **Context Searching Order**:
   1. Current directory's AGENTS.md
   2. `/mamba_ssm/AGENTS.md` (if exists)
   3. Repository root AGENTS.md
   
2. **When Modifying Critical Files**:
   - Always maintain CPU fallback path
   - Add `# VRAM_CRITICAL` comments to sensitive sections
   - Include memory estimation comments:
     ```python
     # [VRAM EST: 320MB] Conv kernel buffer
     self.conv1d = nn.Conv1d(...)
     ```

3. **Documentation Rules**:
   - Add/update docstrings using numpy format
   - Update `README.md#limitations` for new constraints
   - Create visual architecture diagrams in `/assets/`

4. **Testing Philosophy**:
   - Assume no GPU available
   - Mock CUDA dependencies with `unittest.mock`
   - Validate numerical equivalence against CPU reference
   - Include OOM simulation tests

## Hardware Constraints Handling
```markdown
When encountering CUDA-dependent functionality:
1. Implement CPU-compatible fallback first
2. Add feature flag: `if torch.cuda.is_available():`
3. Document limitations in function docstring
4. Create [NO_GPU] test variant
```

**Example Workflow for CUDA Features:**
```python
# In ops module
try:
    from .cuda_ops import fused_mamba_kernel
except ImportError:
    # [NO_GPU] Fallback to CPU implementation
    from .cpu_ops import naive_mamba
```

## Emergency Protocols
**If VRAM Limit Exceeded:**
1. Immediately roll back changes
2. Add `[VRAM_FAIL]` tag to PR
3. Run diagnostic: `python benchmarks/oom_diagnosis.py`
4. Document failure mode in `/docs/oom_cases.md`

# Training Capability Matrix

**Date**: 2026-07-15  
**Purpose**: Feature-by-feature comparison of Ultralytics vs Run Monitor

---

## Capability Comparison

| Capability | Ultralytics | Run Monitor | Status |
|-----------|-------------|-------------|--------|
| **Dataset Creation** | ✓ Full | ✗ | existing-ultralytics |
| **Subprocess Execution** | ✓ | ✓ | duplicate |
| **Process Tree Kill** | ✓ | ✓ | duplicate, reusable |
| **Event Protocol** | ✓ Custom | ✓ JSON | duplicate, incompatible |
| **Progress Tracking** | ✓ CSV | ⚠️ Events | existing-ultralytics |
| **Training Images** | ✓ | ✗ | existing-ultralytics |
| **CPU/Memory Monitor** | ✗ | ✓ | existing-run-monitor |
| **GPU Monitoring** | ✗ | ✓ | existing-run-monitor |
| **Model Export** | ✓ 15 formats | ✗ | existing-ultralytics |
| **Run Persistence** | ✗ | ✓ .trainlens/ | existing-run-monitor |
| **Script Detection** | ✗ | ✓ | existing-run-monitor |
| **Environment Detection** | ✗ | ✓ | existing-run-monitor |
| **Framework Support** | Ultralytics only | Any Python | needs-unification |
| **Annotation Integration** | ✓ Deep | ✗ | existing-ultralytics |

**Legend**: ✓ Full, ⚠️ Partial, ✗ None

---

## Duplicate Implementations (Must Unify)

1. **Process Termination** - Identical taskkill/killpg code
2. **Log Redirection** - Both use QObject+pyqtSignal
3. **Subprocess Management** - Different but overlapping
4. **Event Parsing** - Incompatible protocols

---

*See TRAINING_SYSTEM_AUDIT.md for detailed code locations*

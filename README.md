# Edge-ML Sensor Pipeline (ML-first, no hardware)

A TinyML-style activity classifier: lightweight neural network trained in Python, with inference re-implemented in plain C++ for edge-style execution under tight memory constraints.

## Current Status
 Phase 1 complete:
- Project scaffold created
- Training script added (`train/train_model.py`)
- Model trained/evaluated on synthetic data
- `requirements.txt` generated for reproducibility

## Project Goal
Build a complete edge-style ML pipeline:
1. Train lightweight model in Python  
2. Export learned weights  
3. Run inference in plain C++ with fixed-size arrays  
4. Simulate streaming sensor input and report latency/memory  

## Repository Structure
- `train/` → model training scripts  
- `src/` → C++ inference engine (in progress)  
- `model/` → exported model weights (in progress)  
- `data/` → dataset files (in progress)  
- `results.md` → experiment logs and metrics  

## Run
```bash
source venv/bin/activate
python train/train_model.py
```

## Latest Result
- Accuracy on unseen synthetic test data: **100.0%**
- Classes: `walk`, `sit`, `run`

## Notes
Current accuracy is from intentionally simple synthetic data used to validate the pipeline.
Real sensor/windowed feature data will be added in the next phase for realistic evaluation.

## Next Milestones
- [ ] Replace synthetic data with CSV-based dataset  
- [ ] Add preprocessing + feature scaling  
- [ ] Export model weights (`W1`, `b1`, `W2`, `b2`)  
- [ ] Implement C++ forward pass (`ReLU`, `argmax`)  
- [ ] Benchmark inference latency and memory footprint  

## Learning Outcomes
- Environment isolation with `venv`  
- Reproducible ML setup with `requirements.txt`  
- Basic neural network training/evaluation flow  
- Foundation for Python-training → C++-inference deployment  

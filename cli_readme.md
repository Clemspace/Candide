# Candide CLI - Installation & Usage

## 🎨 Features

- **ASCII Art Splash Screen** with Voltaire's portrait
- **Beautiful Unicode Progress Bars** for training
- **Rich Terminal Output** with colors and formatting
- **Complete Project Scaffolding** with templates
- **System Diagnostics** for dependency checking
- **Component Registry** management

## 📦 Installation

```bash
# Install dependencies
pip install click pyyaml

# Optional: for even richer output
pip install rich

# Make executable
chmod +x candide_cli.py

# Optional: Install globally
sudo cp candide_cli.py /usr/local/bin/candide
```

## 🚀 Quick Start

### 1. Launch with Splash Screen

```bash
./candide_cli.py
```

You'll see Voltaire's ASCII portrait and the Candide banner!

### 2. Initialize a New Project

```bash
./candide_cli.py init my-llm --template llama
cd my-llm
```

**Available templates:**
- `minimal` - Bare-bones embedding + head
- `gpt` - GPT-style transformer (12 layers, 768d)
- `llama` - LLaMA-style (32 layers, 4096d, GQA, RoPE)
- `mamba` - Mamba state-space model

### 3. Train a Model

```bash
./candide_cli.py train configs/base.yaml --gpus 4
```

With options:
```bash
# Dry run (validate only)
./candide_cli.py train configs/base.yaml --dry-run

# Resume from checkpoint
./candide_cli.py train configs/base.yaml --resume checkpoints/step_1000.pt

# Enable profiling
./candide_cli.py train configs/base.yaml --profile

# Distributed training
./candide_cli.py train configs/base.yaml --gpus 8 --nodes 4
```

### 4. Validate Configuration

```bash
# Quick validation
./candide_cli.py validate configs/base.yaml

# Detailed output
./candide_cli.py validate configs/base.yaml --verbose
```

### 5. System Diagnostics

```bash
# Quick check
./candide_cli.py doctor

# Full system scan
./candide_cli.py doctor --full
```

### 6. Component Registry

```bash
# List all components
./candide_cli.py registry list

# List specific category
./candide_cli.py registry list --category attention

# Detailed view
./candide_cli.py registry list --detailed

# Register new component
./candide_cli.py registry add block my_custom_block --path models.custom
```

### 7. Framework Info

```bash
./candide_cli.py info
```

## 🎨 Screenshots (Simulated)

### Splash Screen
```
                    @@@@@@@@@@                    
                 @@@          @@@                 
               @@                @@               
             @@      @@    @@      @@             
            @@       @@    @@       @@            
           @@         @@@@@@         @@           
          @@     @@          @@     @@          
          @@    @  @@      @@  @    @@          
         @@        @@  @@  @@        @@         
         @@          @@@@@@          @@         
         @@       @@@@    @@@@       @@         
          @@     @  @@@@@@@@  @     @@          
          @@        @@@@@@@@        @@          
           @@       @@    ██       @@           
            ██        ████        ██            
             @@       @  @       @@             
               @@    @@@@@@    @@               
                 @@@          @@@                 
                    @@@@@@@@@@                    

 ██████╗ █████╗ ███╗   ██╗██████╗ ██╗██████╗ ███████╗
██╔════╝██╔══██╗████╗  ██║██╔══██╗██║██╔══██╗██╔════╝
██║     ███████║██╔██╗ ██║██║  ██║██║██║  ██║█████╗  
██║     ██╔══██║██║╚██╗██║██║  ██║██║██║  ██║██╔══╝  
╚██████╗██║  ██║██║ ╚████║██████╔╝██║██████╔╝███████╗
 ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝╚═════╝ ╚══════╝

  Framework for Next-Generation Foundation Models
  Configuration-driven • Modality-agnostic • Production-ready
```

### Training Progress
```
🚀 Candide Training
   ──────────────────────────────────────────────────

   Config:      configs/base.yaml
   Resources:   4 GPUs, 1 node

📊 Training Progress

   Steps [████████████████████░░░░░░░░] 65% 00:02:34
   Step   6500 │ Loss: 0.8234 │ LR: 3.00e-04
```

### System Diagnostics
```
🩺 Candide System Diagnostics
   ──────────────────────────────────────────────────

   ✅ Python 3.12.3
   ✅ PyTorch 2.4.0
   ✅ CUDA available (4 GPUs)
      └─ NVIDIA A100-SXM4-80GB

✅ All checks passed (3/3)
```

## 🎯 Command Reference

| Command | Description |
|---------|-------------|
| `candide init <name>` | Create new project |
| `candide train <config>` | Start training |
| `candide validate <config>` | Validate configuration |
| `candide doctor` | System diagnostics |
| `candide registry list` | List components |
| `candide registry add` | Register component |
| `candide info` | Framework info |
| `candide --help` | Show all commands |

## ⚙️ Configuration File Format

```yaml
model:
  preset: llama  # or 'gpt', or use custom graph
  vocab_size: 50000
  d_model: 4096
  n_layers: 32
  n_heads: 32

training:
  max_steps: 100000
  lr: 3e-4
  warmup_steps: 2000
  batch_size: 512
  
  loss:
    type: cross_entropy
  
  optimizer:
    type: adamw
    betas: [0.9, 0.95]
    weight_decay: 0.1
```

## 🎨 Color Scheme

- **Cyan/Blue** - Headers and titles
- **Green** - Success messages
- **Yellow** - Warnings
- **Red** - Errors
- **Magenta** - Special items (checkpoints, etc.)
- **Dim White** - Supplementary info

## 🚀 Next Steps

1. **Copy to your project:**
   ```bash
   cp candide_cli.py ~/candide_cracked/candide1.0/
   cd ~/candide_cracked/candide1.0
   chmod +x candide_cli.py
   ```

2. **Test it out:**
   ```bash
   ./candide_cli.py  # See the splash!
   ./candide_cli.py init test-project --template llama
   ```

3. **Integrate with your codebase:**
   - Import your actual components
   - Wire up real training logic
   - Add model loading/saving
   - Connect to your registry

## 📝 Notes

- The CLI uses `click` for argument parsing
- Progress bars use Unicode box-drawing characters
- Colors require terminal support (works on most modern terminals)
- ASCII art optimized for 80-column terminals
- All functionality is modular and easy to extend


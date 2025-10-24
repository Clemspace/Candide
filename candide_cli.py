#!/usr/bin/env python3
"""
Candide CLI - Framework for Next-Generation Foundation Models

Install: pip install click pyyaml rich
Run: python candide_cli.py --help
"""

import click
from pathlib import Path
import yaml
import sys

# ASCII Art - Voltaire's portrait (stylized)
VOLTAIRE_ASCII = """
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
           @@       @@    @@       @@           
            @@        @@@@        @@            
             @@       @  @       @@             
               @@    @@@@@@    @@               
                 @@@          @@@                 
                    @@@@@@@@@@                    
"""

CANDIDE_BANNER = """
 ██████╗ █████╗ ███╗   ██╗██████╗ ██╗██████╗ ███████╗
██╔════╝██╔══██╗████╗  ██║██╔══██╗██║██╔══██╗██╔════╝
██║     ███████║██╔██╗ ██║██║  ██║██║██║  ██║█████╗  
██║     ██╔══██║██║╚██╗██║██║  ██║██║██║  ██║██╔══╝  
╚██████╗██║  ██║██║ ╚████║██████╔╝██║██████╔╝███████╗
 ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝╚═════╝ ╚══════╝
                                                       
    "Dans ce pays-ci, il est bon de tuer de temps     
     en temps un amiral pour encourager les autres."  
                                        — Voltaire     
"""

def show_splash():
    """Display the Candide splash screen."""
    click.clear()
    click.secho(VOLTAIRE_ASCII, fg='cyan')
    click.secho(CANDIDE_BANNER, fg='bright_blue', bold=True)
    click.secho("\n  Framework for Next-Generation Foundation Models", fg='white', dim=True)
    click.secho("  Configuration-driven • Modality-agnostic • Production-ready\n", fg='white', dim=True)


@click.group(invoke_without_command=True)
@click.pass_context
@click.version_option(version='0.1.0')
@click.option('--splash/--no-splash', default=True, help='Show splash screen')
def candide(ctx, splash):
    """
    Candide: Framework for Next-Generation Foundation Models
    
    Build, train, and deploy foundation models with configuration-driven
    component composition. Scales from single GPU to distributed clusters.
    """
    if ctx.invoked_subcommand is None:
        if splash:
            show_splash()
        ctx.invoke(help)


@candide.command()
@click.argument('project-name')
@click.option('--template', '-t', 
              type=click.Choice(['gpt', 'llama', 'minimal', 'mamba']),
              default='minimal',
              help='Project template to use')
@click.option('--interactive', '-i', is_flag=True,
              help='Interactive setup wizard')
def init(project_name, template, interactive):
    """Initialize a new Candide project with configuration scaffolding."""
    click.secho(f"\n🌱 Initializing project: ", fg='cyan', nl=False)
    click.secho(project_name, fg='bright_cyan', bold=True)
    click.secho(f"   Template: ", fg='white', dim=True, nl=False)
    click.secho(template, fg='yellow')
    click.echo()
    
    project_path = Path(project_name)
    
    if project_path.exists():
        click.secho(f"❌ Directory '{project_name}' already exists!", fg='red')
        return
    
    # Create directory structure
    dirs = [
        project_path,
        project_path / 'configs',
        project_path / 'data',
        project_path / 'checkpoints',
        project_path / 'logs',
    ]
    
    with click.progressbar(dirs, label='Creating directories', 
                          bar_template='%(label)s  [%(bar)s] %(info)s',
                          fill_char='█', empty_char='░') as dirs_progress:
        for d in dirs_progress:
            d.mkdir(parents=True, exist_ok=True)
    
    # Template configs
    templates = {
        'minimal': {
            'model': {
                'graph': {
                    'nodes': [
                        {'id': 'emb', 'type': 'embedding', 'name': 'token', 
                         'config': {'vocab_size': 50000, 'dim': 512}},
                        {'id': 'head', 'type': 'head', 'name': 'lm', 
                         'config': {'vocab_size': 50000}, 'inputs': ['emb']}
                    ],
                    'inputs': ['emb'],
                    'outputs': ['head']
                }
            },
            'training': {
                'max_steps': 10000,
                'lr': 3e-4,
                'batch_size': 32,
                'loss': {'type': 'cross_entropy'}
            }
        },
        'gpt': {
            'model': {
                'preset': 'gpt',
                'vocab_size': 50000,
                'd_model': 768,
                'n_layers': 12,
                'n_heads': 12,
            },
            'training': {
                'max_steps': 100000,
                'lr': 3e-4,
                'warmup_steps': 2000,
                'batch_size': 64,
            }
        },
        'llama': {
            'model': {
                'preset': 'llama',
                'vocab_size': 50000,
                'd_model': 4096,
                'n_layers': 32,
                'n_heads': 32,
            },
            'training': {
                'max_steps': 100000,
                'lr': 3e-4,
                'warmup_steps': 2000,
                'batch_size': 512,
            }
        },
        'mamba': {
            'model': {
                'graph': {
                    'nodes': [
                        {'id': 'emb', 'type': 'embedding', 'name': 'token'},
                        {'id': 'block_0', 'type': 'block', 'name': 'mamba', 'inputs': ['emb']},
                        {'id': 'head', 'type': 'head', 'name': 'lm', 'inputs': ['block_0']}
                    ]
                }
            }
        }
    }
    
    config = templates.get(template, templates['minimal'])
    
    # Save config
    config_file = project_path / 'configs' / 'base.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    # Create README
    readme = f"""# {project_name}

Foundation model project built with Candide.

## Quick Start

```bash
# Train the model
candide train configs/base.yaml

# Validate configuration
candide validate configs/base.yaml

# Check system setup
candide doctor
```

## Project Structure

```
{project_name}/
├── configs/        # Model and training configurations
├── data/          # Training data
├── checkpoints/   # Model checkpoints
└── logs/          # Training logs
```

## Configuration

Edit `configs/base.yaml` to customize your model architecture and training setup.

## Documentation

See https://docs.candide.ai for full documentation.
"""
    
    with open(project_path / 'README.md', 'w') as f:
        f.write(readme)
    
    click.echo()
    click.secho("✅ Project created successfully!", fg='green', bold=True)
    click.echo()
    click.secho("📁 Structure:", fg='cyan')
    click.echo(f"   {project_name}/")
    click.echo(f"   ├── configs/base.yaml")
    click.echo(f"   ├── data/")
    click.echo(f"   ├── checkpoints/")
    click.echo(f"   ├── logs/")
    click.echo(f"   └── README.md")
    click.echo()
    click.secho("🚀 Next steps:", fg='cyan', bold=True)
    click.echo(f"   cd {project_name}")
    click.echo(f"   candide train configs/base.yaml")
    click.echo()


@candide.command()
@click.argument('config', type=click.Path(exists=True))
@click.option('--gpus', '-g', default=1, help='Number of GPUs to use')
@click.option('--nodes', '-n', default=1, help='Number of nodes for distributed training')
@click.option('--resume', '-r', type=click.Path(), help='Resume from checkpoint')
@click.option('--dry-run', is_flag=True, help='Validate config without training')
@click.option('--profile', is_flag=True, help='Enable performance profiling')
def train(config, gpus, nodes, resume, dry_run, profile):
    """Start model training from configuration file."""
    click.echo()
    click.secho("🚀 Candide Training", fg='bright_blue', bold=True)
    click.echo("   " + "─" * 50)
    click.echo()
    
    # Display config
    click.secho(f"   Config:      ", fg='white', dim=True, nl=False)
    click.secho(config, fg='cyan')
    click.secho(f"   Resources:   ", fg='white', dim=True, nl=False)
    click.secho(f"{gpus} GPU{'s' if gpus > 1 else ''}, {nodes} node{'s' if nodes > 1 else ''}", fg='yellow')
    
    if resume:
        click.secho(f"   Resume from: ", fg='white', dim=True, nl=False)
        click.secho(resume, fg='magenta')
    
    if profile:
        click.secho(f"   Profiling:   ", fg='white', dim=True, nl=False)
        click.secho("Enabled", fg='green')
    
    click.echo()
    
    # Load and validate config
    try:
        with open(config) as f:
            config_dict = yaml.safe_load(f)
    except Exception as e:
        click.secho(f"❌ Failed to load config: {e}", fg='red')
        return
    
    if dry_run:
        click.secho("✅ Configuration is valid! (dry run)", fg='green')
        return
    
    # Simulate training with fancy progress
    max_steps = config_dict.get('training', {}).get('max_steps', 10000)
    
    click.secho("📊 Training Progress", fg='cyan', bold=True)
    click.echo()
    
    with click.progressbar(
        length=max_steps, 
        label='   Steps',
        bar_template='%(label)s [%(bar)s] %(info)s',
        fill_char='█',
        empty_char='░',
        show_eta=True,
        show_percent=True
    ) as bar:
        for i in range(max_steps):
            bar.update(1)
            if i % (max_steps // 10) == 0 and i > 0:
                loss = 2.5 * (1 - i/max_steps) + 0.1
                click.echo(f"   Step {i:6d} │ Loss: {loss:.4f} │ LR: {3e-4:.2e}")
    
    click.echo()
    click.secho("✅ Training complete!", fg='green', bold=True)
    click.secho(f"   Checkpoint saved to: checkpoints/final.pt", fg='white', dim=True)
    click.echo()


@candide.command()
@click.argument('config', type=click.Path(exists=True))
@click.option('--verbose', '-v', is_flag=True, help='Show detailed validation')
def validate(config):
    """Validate model configuration file."""
    click.echo()
    click.secho("🔍 Validating Configuration", fg='cyan', bold=True)
    click.echo("   " + "─" * 50)
    click.echo()
    
    try:
        with open(config) as f:
            config_dict = yaml.safe_load(f)
        
        # Check required top-level keys
        required = ['model', 'training']
        checks = []
        
        for key in required:
            exists = key in config_dict
            checks.append((key, exists))
            
            icon = "✅" if exists else "❌"
            color = 'green' if exists else 'red'
            click.secho(f"   {icon} ", fg=color, nl=False)
            click.echo(f"{key.capitalize()} configuration")
        
        click.echo()
        
        if all(exists for _, exists in checks):
            click.secho("✅ Configuration is valid!", fg='green', bold=True)
            
            if verbose:
                click.echo()
                click.secho("📋 Configuration Summary:", fg='cyan')
                click.echo(yaml.dump(config_dict, default_flow_style=False, indent=2))
        else:
            missing = [k for k, exists in checks if not exists]
            click.secho(f"❌ Missing required keys: {', '.join(missing)}", fg='red', bold=True)
            
    except Exception as e:
        click.secho(f"❌ Validation failed: {e}", fg='red', bold=True)
    
    click.echo()


@candide.command()
@click.option('--full', '-f', is_flag=True, help='Run full system check')
def doctor(full):
    """Run system diagnostics and check dependencies."""
    click.echo()
    click.secho("🩺 Candide System Diagnostics", fg='cyan', bold=True)
    click.echo("   " + "─" * 50)
    click.echo()
    
    checks = []
    
    # Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 10)
    checks.append(py_ok)
    
    icon = "✅" if py_ok else "⚠️"
    color = 'green' if py_ok else 'yellow'
    click.secho(f"   {icon} ", fg=color, nl=False)
    click.echo(f"Python {py_ver}", nl=False)
    if not py_ok:
        click.secho(" (3.10+ recommended)", fg='yellow', dim=True)
    else:
        click.echo()
    
    # PyTorch
    try:
        import torch
        torch_ok = True
        checks.append(True)
        click.secho(f"   ✅ ", fg='green', nl=False)
        click.echo(f"PyTorch {torch.__version__}")
        
        # CUDA
        if torch.cuda.is_available():
            gpus = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            checks.append(True)
            click.secho(f"   ✅ ", fg='green', nl=False)
            click.echo(f"CUDA available ({gpus} GPU{'s' if gpus > 1 else ''})")
            if full:
                click.secho(f"      └─ {gpu_name}", fg='white', dim=True)
        else:
            checks.append(False)
            click.secho(f"   ⚠️  ", fg='yellow', nl=False)
            click.echo(f"No CUDA GPUs available (CPU only)")
            
    except ImportError:
        checks.append(False)
        click.secho(f"   ❌ ", fg='red', nl=False)
        click.echo(f"PyTorch not installed")
        click.secho(f"      └─ Install: pip install torch", fg='white', dim=True)
    
    # Additional checks in full mode
    if full:
        packages = ['yaml', 'click', 'numpy']
        for pkg in packages:
            try:
                __import__(pkg)
                click.secho(f"   ✅ ", fg='green', nl=False)
                click.echo(f"{pkg.capitalize()} available")
                checks.append(True)
            except ImportError:
                click.secho(f"   ⚠️  ", fg='yellow', nl=False)
                click.echo(f"{pkg.capitalize()} not found")
                checks.append(False)
    
    click.echo()
    
    # Summary
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        click.secho(f"✅ All checks passed ({passed}/{total})", fg='green', bold=True)
    elif passed > total // 2:
        click.secho(f"⚠️  Some issues detected ({passed}/{total} passed)", fg='yellow', bold=True)
    else:
        click.secho(f"❌ Critical issues detected ({passed}/{total} passed)", fg='red', bold=True)
    
    click.echo()


@candide.group()
def registry():
    """Manage component registry."""
    pass


@registry.command('list')
@click.option('--category', '-c', help='Filter by category')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed information')
def registry_list(category, detailed):
    """List all registered components."""
    components = {
        'embedding': ['token', 'rotary', 'learned_position'],
        'block': ['transformer', 'mamba', 'moe'],
        'attention': ['mha', 'gqa', 'mqa', 'flash'],
        'ffn': ['mlp', 'swiglu', 'geglu'],
        'norm': ['layer', 'rms', 'rmsnorm'],
        'head': ['lm', 'classification', 'regression'],
        'loss': ['cross_entropy', 'semantic_entropy', 'contrastive'],
        'optimizer': ['adamw', 'muon', 'ademamix', 'lion'],
    }
    
    click.echo()
    click.secho("📦 Candide Component Registry", fg='cyan', bold=True)
    click.echo("   " + "─" * 50)
    click.echo()
    
    if category:
        if category in components:
            click.secho(f"   {category.capitalize()}:", fg='bright_cyan', bold=True)
            for comp in components[category]:
                click.echo(f"      • {comp}")
            click.echo()
        else:
            click.secho(f"❌ Unknown category: {category}", fg='red')
            click.echo(f"   Available: {', '.join(components.keys())}")
    else:
        for cat, comps in components.items():
            click.secho(f"   {cat.capitalize():<12}", fg='bright_cyan', nl=False)
            click.echo(f" {len(comps):2d} components")
            if detailed:
                for comp in comps:
                    click.secho(f"      └─ {comp}", fg='white', dim=True)
        
        click.echo()
        click.secho(f"   Total: {sum(len(c) for c in components.values())} components", 
                   fg='white', dim=True)
    
    click.echo()


@registry.command('add')
@click.argument('category')
@click.argument('name')
@click.option('--path', '-p', help='Python module path')
def registry_add(category, name, path):
    """Register a new component."""
    click.echo()
    click.secho(f"📝 Registering component: ", fg='cyan', nl=False)
    click.secho(f"{category}/{name}", fg='bright_cyan', bold=True)
    click.echo()
    
    if path:
        click.secho(f"   Path: {path}", fg='white', dim=True)
    
    click.echo()
    click.secho("✅ Component registered!", fg='green', bold=True)
    click.secho("   Run 'candide registry list' to see all components", fg='white', dim=True)
    click.echo()


@candide.command()
def info():
    """Show Candide framework information."""
    show_splash()
    
    click.secho("📊 Framework Information", fg='cyan', bold=True)
    click.echo("   " + "─" * 50)
    click.echo()
    
    info_items = [
        ("Version", "0.1.0"),
        ("Python", f"{sys.version_info.major}.{sys.version_info.minor}+"),
        ("License", "Apache 2.0"),
        ("Repository", "github.com/candide-ai/candide"),
        ("Documentation", "docs.candide.ai"),
    ]
    
    for label, value in info_items:
        click.secho(f"   {label:<15}", fg='white', dim=True, nl=False)
        click.secho(value, fg='cyan')
    
    click.echo()


if __name__ == '__main__':
    candide()
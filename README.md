# Evolution Simulator 🌱🧬

A custom individual-creature focused evolutionary simulator with advanced genetics and optimized performance

## Simulation Preview 🔍

<div style="display: flex; overflow-x: auto; gap: 16px; padding: 16px 0;">
  <div style="flex: 0 0 auto; min-width: 300px; text-align: left;">
    <img src="./images/1.png" style="height: 200px; border-radius: 8px;">
    <img src="./images/2.png" style="height: 200px; border-radius: 8px;">
    <img src="./images/3.png" style="height: 200px; border-radius: 8px;">
    <p> </p>
  </div>
</div>


## Key Features 🔑

### Core Simulation
<br>🕒 **4-Season Time System** - Minutes=seasons, seconds=days with automatic food blooms<br>
🌾 **Dynamic Food Ecosystem** - Seasonal respawning + daily sprinkles + cluster-based generation<br>
👀 **Selectable Creatures** - Click any organism to view real-time stats and genetic makeup<br>
🧠 **Smart Creature AI** - Food/mate seeking behaviors with FOV-based ray casting

### Genetic System
<br>🧬 **True DNA Simulation** - Genome/Gene class system with allele inheritance<br>
🎲 **Mutations** - Configurable mutation rates (0.1-5%) with dominance factors<br>
💞 **Sexual Reproduction** - Crossover breeding with trait combination<br>
🌈 **Trait Expression** - Physical traits derived from genetic combinations<br>
📜 **Epigenetic Effects** - Environmental factors influence gene activation

### Technical Highlights
<br>🚀 **Spatial Partitioning** - 150px grid system for 10x faster collision detection<br>
🖥️ **OpenGL Rendering** - Batched texture rendering with GraphicsRenderer class<br>
🧵 **Multithreaded Processing** - Async food generation and parallel ray casting<br>
💾 **Persistent World** - Auto-saves progress between sessions

### Advanced Mechanics
<br>⚡ **Energy System** - Metabolism/size/speed tradeoffs with hunger dynamics<br>
👓 **Vision System** - Customizable FOV (110-140°) and sight range (130-160px)<br>
🍎 **Complex Food** - Branching food structures with variable energy values<br>
📊 **Real-time UI** - Toggle debug overlays/FOV + leaderboard rankings

### Creature Details
<br>🔠 **Procedural Names** - Linguistically plausible name generation<br>
⏳ **Aging System** - Lifespan (80-90s) with age-related stats<br>
🎯 **Behavior States** - Food seeking/mating/wandering modes<br>
👥 **Social Interactions** - Energy-based mating rituals

## Roadmap 🗺️

### Next Major Features
<br>🦖 **Species System** - Carnivore/herbivore speciation<br>
🌐 **Expanded Food Web** - Multiple food types with nutritional values<br>
🧬 **Gene Regulation** - Activator/repressor gene networks

### Genetic Expansion
<br>🧩 **Pleiotropic Genes** - Single genes affecting multiple traits<br>
🎭 **Polygenic Traits** - Combined gene effects on characteristics<br>
⚠️ **Genetic Disorders** - Harmful mutation possibilities<br>
🌡️ **Environmental DNA** - Temperature-dependent gene expression

### Technical Upgrades
<br>🌐 **Distributed Computing** - Offload simulation to GPU<br>
📈 **Data Tracking** - Generational lineage graphs<br>
🎮 **Interactive Evolution** - Player-directed breeding

## Installation ⚙️
Use the run.bat for easy install and run. It will do the following inside of your current directory:
```bash
git clone https://github.com/yourusername/evolution-simulator.git
cd evolution-simulator
pip install -r requirements.txt
python main.py
```

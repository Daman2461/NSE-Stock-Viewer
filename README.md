#  Multi-Agent Reinforcement Learning framework to optimize derivative pricing models for complex options.

view website at https://nse-stock.streamlit.app
# NSE-Stock-Viewer

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nse-stock.streamlit.app)

## Overview

**NSE-Stock-Viewer** is a Multi-Agent Reinforcement Learning (MARL) framework to optimize derivative pricing models for complex options. The project leverages machine learning and agent-based modeling to analyze and price financial derivatives using real-world data from the National Stock Exchange (NSE).

- **Project Homepage:** [https://nse-stock.streamlit.app](https://nse-stock.streamlit.app)
- **Repository:** [Daman2461/NSE-Stock-Viewer](https://github.com/Daman2461/NSE-Stock-Viewer)

## Features

- Multi-agent reinforcement learning for financial modeling
- Derivative options pricing using advanced ML techniques
- Interactive dashboards and visualization (Streamlit app)
- Pre-trained models and policy checkpoints for reproducibility

## Getting Started

### Prerequisites

- Python 3.8+
- Recommended: virtualenv or conda

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Daman2461/NSE-Stock-Viewer.git
   cd NSE-Stock-Viewer
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Launch the Streamlit Dashboard:
```bash
streamlit run stock_app.py
```

Or, visit the hosted version: [nse-stock.streamlit.app](https://nse-stock.streamlit.app)

## Repository Structure

- `stock_app.py` : Main Streamlit application
- `Cascading-2.ipynb`, `MARL_options.ipynb` : Research and model development notebooks
- `requirements.txt` : Python dependencies
- `model_level_*.joblib` : Pre-trained model files
- `policy/`, `q_network_checkpoint-1/` : Saved policies and checkpoints

## Example Usage

Adjust or explore the Streamlit app to:
- Visualize option pricing dynamics
- Experiment with different agent behaviors
- Analyze model predictions on real NSE data

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

## License

_No license information specified. Please add a LICENSE file if you wish to define usage rights._

## Acknowledgements

- Inspired by financial modeling and reinforcement learning research communities.
- Data sourced from the National Stock Exchange (NSE).

---

_This README was auto-generated. For more files or details, visit the [repository root](https://github.com/Daman2461/NSE-Stock-Viewer/tree/main/)._

# ByteFight 2025 – Reinforcement‑Learning Agent

> **Competition:** [Georgia Tech ByteFight 2025](https://bytefight.org)  •  **Final Rank:** **8ᵗʰ**

ByteFight 2025 is a head‑to‑head arena inspired by the classic *Snake* but adds variable maps, multi-move movements and direct combat with the goal to survive longer than you're oppponent. Each bot receives the complete board state every tick and must attempt to stay alive for as long as possible without running out of time.

This repository contains my reinforcement‑learning (RL) solution that climbed to the top‑10 after just several days of training.

---

## Key Ideas

- **Custom Gymnasium env** – precisely mirrors the ByteFight server API so policies can be trained off‑line and dropped straight into competition.
- **PPO with Stable Baselines3** – convolutional encoder + MLP policy trained on 2 M self‑play steps.
- **Pure‑NumPy inference** – competition VMs allow only builtin Python and NumPy. I therefore re‑implemented the trained CNN layers in NumPy, then imported SB3‑trained weights.
- **Offence‑aware reward shaping** – combines territory control, health conservation, and opponent proximity to teach both evasive maneuvers and aggressive traps.

---

## Overcoming Server Constraints

ByteFight finals ran on stripped‑down Docker images with **no** PyTorch, Gymnasium, or Stable‑Baselines3.  The workaround:

1. **Weight Extraction** – Trained custom network using stablebaseline and then extracting paremeters after training
2. **NumPy CNN** – implement conv/ELU/max‑pool/dense layers in Numpy
3. **Load & Run** – load parameters on competition servers before using custom network to perform inference/

---

## Results

- 8ᵗʰ place overall (single‑elimination finals).
- High‑scoring fully RL‑based entrant; other top bots relied on hand‑crafted heuristics.

---

## Acknowledgements

- **Georgia Tech ByteFight** – competition organisers.
- **Stable‑Baselines3** team for an excellent RL toolkit.

---

## License

MIT – see `LICENSE` for details. Feel free to fork, learn! 🚀


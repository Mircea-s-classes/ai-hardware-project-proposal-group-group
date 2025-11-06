# University of Virginia
## Department of Electrical and Computer Engineering

**Course:** ECE 4332 / ECE 6332 — AI Hardware Design and Implementation  
**Semester:** Fall 2025  
**Proposal Deadline:** November 5, 2025 — 11:59 PM  
**Submission:** Upload to Canvas (PDF) and to GitHub (`/docs` folder)

---

# AI Hardware Project Proposal Template

## 1. Project Title
Name of the Team: group-group

List of students in the team
1. Claire Chiang
2. Zourong Jiang
3. Sonia Aung
4. Allison Lampe

Provide a clear and concise title for your project: Temperature Forecasting with TinyML 

## 2. Platform Selection
We will use TinyML because weather forcasting can already done adequately on the cloud, but there could be utility in being able to estimate future temperature just with a device. TinyML allows us to, in this case, get real time weather predictions in any location.


## 3. Problem Definition
We aim to design a low-power, always-on system that can operate entirely offline, without reliance on cloud-based weather data. Our system will predict whether the outside temperature will increase or decrease in the next hour based on barometric, humidity, and temperature data taken on a constrained microcontroller platform.

## 4. Technical Objectives
List 3–5 measurable objectives with quantitative targets when possible.

Model Footprint: The final quantized (INT8) TensorFlow Lite model must have a RAM footprint (Tensor Arena) of less than 100 KB, to successfully run on the Arduino Nano's 256KB of RAM.
Predictive accuracy: The model must achieve a predictive accuracy of at least 70% on a test dataset.
Latency: Making one prediction on the Arduino hardware must complete in less than 1000 milliseconds.

1. Accuracy - We target about 70% accuracy because of the simplicity predicting weather from a few parameters. This value may be updated once we analyze preliminary data.
2. Scalability - We will have access to barometric, temperature, and humidity data. We will try every combination of these to understand how the system responds to multiple features and more data.
3. Memory efficiency – We will try to keep the model under 50 kB of RAM and 200 kB of flash, using less than 20% of the memory on the Nano 33 BLE Sense.

## 5. Methodology
Describe your planned approach: hardware setup, software tools, model design, performance metrics, and validation strategy.

## 6. Expected Deliverables
List tangible outputs: working demo, GitHub repository, documentation, presentation slides, and final report.

## 7. Team Responsibilities
List each member’s main role.

| Name | Role | Responsibilities |
|------|------|------------------|
| [Student A] | Team Lead | Coordination, documentation |
| [Student B] | Hardware | Setup, integration |
| [Student C] | Software | Model training, inference |
| [Student D] | Evaluation | Testing, benchmarking |

## 8. Timeline and Milestones
Provide expected milestones:

| Week | Milestone | Deliverable |
|------|------------|-------------|
| 2 | Proposal | PDF + GitHub submission |
| 4 | Midterm presentation | Slides, preliminary results |
| 6 | Integration & testing | Working prototype |
| Dec. 18 | Final presentation | Report, demo, GitHub archive |

## 9. Resources Required
List special hardware, datasets, or compute access needed.

## 10. References
Include relevant papers, repositories, and documentation.

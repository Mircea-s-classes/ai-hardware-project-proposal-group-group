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
1. Accuracy - We target about 70% accuracy because of the simplicity predicting weather from a few parameters. This value may be updated once we analyze preliminary data.
2. Scalability - We will have access to barometric, temperature, and humidity data. We will try every combination of these to understand how the system responds to multiple features and more data.
3. Memory efficiency – We will try to keep the model under 50 kB of RAM and 200 kB of flash, using less than 20% of the memory on the Nano 33 BLE Sense.

## 5. Methodology
Hardware Setup:
Arduino Nano 33 BLE Sense for sensing temperature, humidity, and barometric pressure.

Software Tools:
Arduino IDE for sensor interfacing.
Python / TensorFlow for offline model training and quantization.
TensorFlow Lite Micro to deploy the model on the Nano.

Model Design:
Train on historical weather data from meteostat.net to predict temperature trend (up/down) based on humidity, barometric, and current temperature data.

Performance Metrics:
Accuracy on local test data (~70% target).
Memory usage (<50 kB RAM, <200 kB flash).

Validation Strategy:
Test on a week of local sensor data collected from the Nano.

## 6. Expected Deliverables
List tangible outputs: working demo, GitHub repository, documentation, presentation slides, and final report.

## 7. Team Responsibilities
List each member’s main role.

| Name | Role | Responsibilities |
|------|------|------------------|
| Sonia Aung | Team Lead | Coordination, documentation |
| Zourong Jiang | Hardware | Setup, integration |
| Claire Chiang | Software | Model training, inference |
| Allison Lampe | Evaluation | Data Organization, Testing |

## 8. Timeline and Milestones
Provide expected milestones:

| Week | Milestone | Deliverable |
|------|------------|-------------|
| 2 | Proposal | PDF + GitHub submission |
| 4 | Midterm presentation | Slides, preliminary results |
| 6 | Integration & testing | Working prototype |
| Dec. 18 | Final presentation | Report, demo, GitHub archive |

## 9. Resources Required
Arduino Nano 33 BLE Sense, weather data from meteostat.net.

## 10. References
Include relevant papers, repositories, and documentation.

# GEO_Satellite_RA

## Title
《Flexible Resource Management in High-throughput Satellite Communication Systems: A Two-stage Machine Learning Framework》

## Abstract
With digitization and globalization in the era of 5G and beyond, research on high-throughput satellites (HTS) to increase communication capacity and improve flexibility is becoming essential. To achieve efficient resource utilization and dynamic traffic demand matching, the multi-dimensional resource management (MDRM) problem of the HTS communication system has been studied in this paper. Since the MDRM problem is a non-convex mixed integer problem, we decompose it into two tractable sub-problems. First, the beam-domain resource configuration problem is formed to enable ondemand coverage. Next, the user-domain resource allocation problem is modeled to enable on-demand communication. Considering the two-domain optimization problem, a two-stage framework is developed based on the combination of self-supervised learning and deep reinforcement learning. Specifically, in the first stage, a maximum co-channel interference based self-supervised learning method is proposed to perform traffic demand matching through demand awareness. In the second stage, a soft frequency reuse based proximal policy optimization approach is presented to further increase the system capacity through interference coordination. The simulation results demonstrate that our proposed two-stage algorithm outperforms the benchmark schemes in terms of spectrum efficiency and demand satisfaction.

## Contributions
(1) We decompose the original optimization problem into the beam domain and user domain, and we build a two-stage algorithm to approximate the optimal MDRM scheme.  
(2) In the beam domain, an SSL algorithm is proposed to solve the overlapped edge-band constrained resource configuration problem, with the significant advantages of reducing time and computation costs.  
(3) In the user domain, an SFR-based DRL method is proposed to solve the orthogonal edgeband constrained resource allocation problem, which has the advantage of dramatically reducing the action space and exploration time.  
(4) We propose an offlinetraining and online-application paradigm, and the simulation results show that our proposed method significantly outperforms the benchmark schemes in terms of SE, satisfaction and generalization.

## System Model
![images](https://github.com/Zhaodimmmm/GEO_Satellite_RA/blob/master/images/System%20Model.png)

## Algorithm
### Solution 1 (MCI-SSL Algorithm)
We propose a maximum co-channel interference based self-supervised learning [40] (MCI-SSL) method as shown below, and the MCI-SSL algorithm is composed of data collection, network structure design and parameter optimization.
![images](https://github.com/Zhaodimmmm/GEO_Satellite_RA/blob/master/images/MCI-SSL%20Algorithm.png)

### Solution 2 (SFR-PPO Algorithm)
Motivated by the idea of SFR, an improved DRL approach, called SFR-enabled proximal policy optimization (PPO) algorithm (SFR-PPO), is proposed to allocate sub-channels for requesting users.
![images](https://github.com/Zhaodimmmm/GEO_Satellite_RA/blob/master/images/SFR-PPO%20Algorithm.png)

## Simulation
### Simulation Parameters
| Parameter | Value | 
| --------- | ----- |
| Satellite height | 35786 km | 
| Number of beams | 12 |
| Number of users | 180 |
| Number of sub-channels | 10 |
| System bandwidth | 500 MHz |
| Sub-channel bandwidth | 50 MHz |
| Total available transmit power | 3000 W |
| Maximum transmit power | 30 dBw |
| Upper limit of EIRP | 84 dBw |
| Maximum transmit antenna gain | 54 dBi |
| Mimimum transmit antenna gain | 24 dBi |
| Half-power beamwidth | 0.310°~9.811° |
| Carrier frequency | 20 GHz |
| Antenna aperture efficiency | 0.5 |
| Antenna aperture diameter | 2.5 m |
| Receiver antenna gain | 40 dBi |
| Noise power density | -199.6 dBw/MHz |
| Nakagami parameter | 5 |
| Average power of the LOS component | 0.279 |
| Average power of the multi-path component | 0.502 |

### Parameters of MCI-SSL
| Parameter | Value | 
| --------- | ----- |
| Training epochs | 500 |
| Time steps | 500 |
| Input layer | 180×5, Tanh |
| Hidden layer | [768, 384, 192, 96, 48, 24], Tanh |
| Output layer | 12, Sigmoid |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Scaling factor | 30 |

### Parameters of SFR_PPO
| Parameter | Value | 
| --------- | ----- |
| Training epochs | 500 |
| Episodes per epoch | 2 |
| Steps per episode | 50 |
| Buffer size | 100 |
| Discount factor | 0.99 |
| Clip ratio | 0.2 |
| Target KL | 0.015 |
| Hidden layer size | [256, 512, 1024, 512, 256] |
| Activation function | Tanh |
| Training iterations per epoch | 80 |
| Learning rate of Actor | 3e-4 |
| Learning rate of Critic | 1e-4 |

## Benchmarks
| Publication/Date | Paper | Algorithm | Code |
| ------ | --------------------- | ----- | ----- |
| NS3 Simulator | [Proportional Fair (PF) Scheduler](https://www.nsnam.org/docs/models/html/lte-design.html#proportional-fair-pf-scheduler) | PF Algorithm | None |
| IEEE Transactions on Wireless Communications / Jan. 2022 | [DBF-based fusion control of transmit power and beam directivity for flexible resource allocation in HTS communication system toward B5G](https://ieeexplore.ieee.org/document/9479792/) | SA Algorithm | None |
| 2020 10th Advanced Satellite Multimedia Systems Conference and the 16th Signal Processing for Space Communications Workshop (ASMS/SPSC) / Dec. 2020 | [Supervised machine learning for power and bandwidth management in VHTS systems](https://ieeexplore.ieee.org/document/9268790) | SL Algorithm | None |

## Reference
| Publication/Date | Paper | Code |
| ------ | --------------------- | ----- |
| 2020 IEEE Aerospace Conference / Aug. 2020 | [Artificial Intelligence Algorithms for Power Allocation in High Throughput Satellites: A Comparison](https://ieeexplore.ieee.org/document/9172682/) | None |
| Intelligent and Converged Networks / Sep. 2021 | [Artificial intelligence for satellite communication: A review](https://ieeexplore.ieee.org/document/9622204/) | None |
| IEEE Transactions on Cognitive Communications and Networking / Mar. 2022 | [Cooperative Multi-Agent Deep Reinforcement Learning for Resource Management in Full Flexible VHTS Systems](https://ieeexplore.ieee.org/document/9448341/) | None |
| IEEE Transactions on Wireless Communications / Jan. 2022 | [DBF-based fusion control of transmit power and beam directivity for flexible resource allocation in HTS communication system toward B5G](https://ieeexplore.ieee.org/document/9479792/) | None |
| 2021 IEEE 6th International Conference on Computer and Communication Systems (ICCCS) / Jun. 2021 | [Deep Reinforcement Learning for Dynamic Bandwidth Allocation in Multi-Beam Satellite Systems](https://ieeexplore.ieee.org/document/9449160) | None |
| IEEE Open Journal of the Communications Society / Apr. 2022 | [Demand and Interference Aware Adaptive Resource Management for High Throughput GEO Satellite Systems](https://ieeexplore.ieee.org/document/9758046/) | None |
| IEEE Transactions on Wireless Communications / Dec. 2021 | [Flexible Resource Optimization for GEO Multibeam Satellite Communication System](https://ieeexplore.ieee.org/document/9460776) | None |
| IEEE Transactions on Communications / Feb. 2020 | [Joint Beamforming Design and Resource Allocation for Terrestrial-Satellite Cooperation System](https://ieeexplore.ieee.org/document/8886590/) | None |
| Electronics / Mar. 2022 | [Machine Learning for Radio Resource Management in Multibeam GEO Satellite Systems](https://www.mdpi.com/2079-9292/11/7/992) | None |
| IEEE Transactions on Wireless Communications / Oct. 2021 | [Machine Learning-Based Resource Allocation in Satellite Networks Supporting Internet of Remote Things](https://ieeexplore.ieee.org/document/9420293/) | None |
| China Communications / Jan. 2022 | [Multi-Objective Deep Reinforcement Learning Based Time-Frequency Resource Allocation for Multi-Beam Satellite Communications](https://ieeexplore.ieee.org/document/9693472/) | None |
| IEEE Transactions on Wireless Communications / Jun. 2015 | [Power Allocation in Multibeam Satellite Systems: A Two-Stage Multi-Objective Optimization](https://ieeexplore.ieee.org/document/7039249/) | None |
| 2020 10th Advanced Satellite Multimedia Systems Conference and the 16th Signal Processing for Space Communications Workshop (ASMS/SPSC) / Dec. 2020 | [Supervised machine learning for power and bandwidth management in VHTS systems](https://ieeexplore.ieee.org/document/9268790) | SL Algorithm | None |
| IEEE Journal on Selected Areas in Communications / Jan. 2021 | [Situation-Aware Resource Allocation for Multi-Dimensional Intelligent Multiple Access: A Proactive Deep Learning Framework](https://ieeexplore.ieee.org/document/9252919) | None |

## Timeline
Submission：June 7, 2022;  
Major Revision: August 9, 2022;  
Submission (revised): October 7, 2022;  
Major Revision: November 19, 2022;  
Submission (revised): January 17, 2023;  

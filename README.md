# Network-Intrusion-Detection
**ABSTARCT**
Nowadays, with the increasing reliance on internet-based services, both client and server 
systems are vulnerable to cyber-attacks initiated by malicious users. To prevent such attacks, a 
Network Intrusion Detection System (IDS) is essential. IDS monitors each incoming network request 
and checks whether it contains normal or malicious (attack) signatures. If attack signatures are 
detected, the system drops the request and logs it for future reference.
In this project, a supervised machine learning-based IDS is implemented using Python, with 
an interactive graphical user interface (GUI) built using Tkinter. The IDS is trained using the NSLKDD dataset, which contains various known attack signatures. Upon receiving new requests, the 
trained model is applied to predict whether the request is normal or represents an attack.
The system utilizes machine learning algorithms such as Support Vector Machine (SVM), 
Random Forest (RF), Decision Tree (DT), and Artificial Neural Network (ANN). Experimental 
results show that RF achieves higher accuracy compared to the other models, especially SVM. To 
enhance performance, feature selection techniques like Correlation-Based and Chi-Square-Based 
selection are applied to remove irrelevant features from the dataset. This significantly reduces dataset 
size, improves processing time, and enhances overall detection accuracy.
The developed IDS processes each request in real-time, allowing only genuine traffic to reach 
the server while blocking and logging malicious traffic. the IDS not only blocks the request but also 
triggers an on-screen pop-up alert to notify the user. This makes the system an effective and practical 
solution for real-time intrusion detection and prevention.

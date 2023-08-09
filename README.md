# TECHNICAL DESIGN PROJECT
# Requirement
## Client Details
INDUSTRY PARTNER NAME
Bondi Labs
INDUSTRY PARTNER CONTACT DETAILS
Bondi Labs is a dynamic company with a history of working in the cutting-edge field of immersive technologies. From simulation training using game engines to remote vision sharing via smart glasses, they aim to address significant business and social issues with forward- thinking solutions. Bondi Labs is looking to expand its capability by providing artificial intelligence and machine learning solutions to their customers.
## Project Description
### Project Title
Automatic Detection of Carcass Contamination
Keywords: Food Safety, E-coli Identification, Machine Learning, Computer Vision, Deep Learning
### Project background & Aims
Beef carcass contaminations may be a direct cause of foodborne diseases and a potential cause for the drug resistance of human pathogenic agents, and consequently, turns this issue into as high risk and zero tolerance issue for food industries. Some techniques, such as pre-washing and “dag” removal, are currently employed which requires a human visual inspection, microbiological culture analysis, bioluminescent ATP-based assays, and antibody-based microbiological tests. However, these methods are labour-intensive and time-consuming. Although some technologies (such as X-ray imaging, Tray and spectrometer test) showed some successes in the detection of contaminations, they couldn’t practically come into the production due to significant delays in processing results and incomparability of required hardware with slaughterhouse configurations.
### Project Objectives
1- Research/literature review report on examining the possible sensor solutions (vision-based camera solutions such as hyperspectral/multispectral camera, thermal camera, IR camera and RGB camera) for maximizing the visibility (contrast) of the contaminates such as ingesta and faeces and milk.
2- Develop ML/CV models for the detection of carcass contaminations such as faecal, ingesta, milk on the carcasses.
### Technical requirements
Research method skill, Programming language (preferably Python), Machine Learning and Computer Vision.

## Project Resources
### Meat contamination background:
• Hot Spot Camera Faecal Detector (https://www.veritide.com/the-technology/)
• Meat quality evaluation by hyperspectral imaging technique
(https://pubmed.ncbi.nlm.nih.gov/22591341/)
• Evaluation of biological contaminants in foods by hyperspectral imaging (https://pubmed.ncbi.nlm.nih.gov/22591341/)
### Machine learning theory:
• A survey on instance segmentation: state of the art (https://link.springer.com/article/10.1007/s13735-020-00195-x)
• Detectron2 Train an Instance Segmentation Model (https://gilberttanner.com/blog/detectron2-train-a-instance-segmentation-model)

# SOLUTION
# Introduction
## Introduce the client
Established in 2014, Bondi Labs is a dynamic company with a history of working in the cutting-edge technology of immersive technologies to enhance the making decision processes. From simulation training using game engines to remote vision sharing via smart glasses, they aim to address significant business and social issues with forward-thinking solutions.

## The problem
Beef carcass contaminations may be a direct cause of foodborne diseases and a potential cause for the drug resistance of human pathogenic agents, and consequently, turns this issue into a high risk and zero tolerance issue for food industries.  The presence of brain or spinal cord material as an inadvertent contaminant of meat may result from stunning livestock, splitting the carcass, or preparing advanced meat recovery (AMR) products from the vertebral column (G. R. Schmidt et al. 2001).

Providing clean and safe products free of contaminants to the public is a core concern and challenge of Australia’s agricultural and food industries. However, during the process of slaughter, stock (and therefore our food supply) is exposed to several major sources of contamination. Bell (1996) concludes that contamination is a result of direct contact between carcasses and faeces, or indirect contact via a surface which is been exposed to both the carcass and faeces (often knives and removed animal hides). As detailed by Bell, the hock, inside leg, bung and flank are the parts of carcasses most likely to suffer faeces contamination, with carcass skinning and evisceration and holding pens are thought to be the likely sources of contamination.

To protect the community from infection, the industry has a ‘zero-tolerance’ threshold for ingesta and faecal contamination – that is, meat cannot be processed before all observed instances of these contaminations have been removed.

## The current process
Some techniques, such as pre-washing and “dag” removal, are currently employed which requires a human visual inspection, microbiological culture analysis, bioluminescent ATP-based assays, and antibody-based microbiological tests. However, these methods are labour-intensive and time-consuming.

Although some technologies (such as X-ray imaging, Tray and spectrometer test) showed some successes in the detection of contaminations, they couldn’t practically come into the production due to significant delays in processing results and incomparability of required hardware with slaughterhouse configurations.
Also as detailed by Casey, Ramussen and Petrich (1999), the current process for identifying contaminants is manual. Inspecting carcasses visually is a time-consuming process as one inspector needs to thoroughly examine a single carcass at a time in difficult conditions. This makes the process expensive and creates potential bottlenecks in production lines.

More concerning is that these methods are prone to inaccuracies. Inspectors often work in rooms with low and variable light, cold temperatures and significant sensory noise (including smells and sounds). These conditions make it difficult for inspectors to maintain accuracy in their work, and as discussed by Park and Chen (2001), this process is prone to inspector-to-inspector variation.

## Project opportunity
Bondi Labs is looking to expand its capability by providing artificial intelligence and machine learning solutions to their customers. And they have identified an opportunity to develop then deploy the equivalent hardware and machine learning techniques to the dangerous contaminants detecting process which is to apply on the beef carcasses that help the beef supplier to deliver better-qualified beef.

It also to provide the adoptions with the industry standards with a solution that providing both kind of economic, production and health benefits for producer and consumer that reducing the costs, removing the bottlenecks, saving time and protecting people health by using less contamination in beef.

## Project scope
As the proof-of-concept one, this project is for researching, proposing and trying to implementing the selected approach to demonstrate the effectiveness and efficiencies of it in solving the problem.
So learning topics will be about:
•	Standard techniques for authentication of food products and its problems
•	Sensors and cameras which can be used in detecting the contamimations, especially about the HSI (hyperspectral) and MSI (multispectral) imaging
•	Image classification, object localization,  semantic segmentation and instance segmentation
•	Small objects detection algorithms
•	Increase performance and accuracy when detecting small objects
•	Sufficiency in deployment of the model to production

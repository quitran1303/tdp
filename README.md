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

# A brief review of existing work
## Brief of liturature review
For beginning the research, we have gone through many papers and below are the findings for the methods, software framework and hardware components that being used in other RnDs:
 
There are some near-standard and standard techniques using for authentication the food products especially in idetifying the levels of adulteration of meat types and their problems in large-scale deployment to industrial production lines. The current standard approaches (for example DNA (Ballin 2010), loop-mediated isothermal amplification (Abdulmawjood et al. 2014), and liquid chromatography (Chou et al. 2007) and protein-based assays, triacylglycerol analysis) which are using nowadays for authentic the food products, especially effective in identification the levels of adulteration of meat types (ie. lamb adulterated with pork, duck aduterated with beef). However, these approaches often require sampling including extracting proteins and DNA so they are expensive, laborious and technically demanding and so cannot be aligned in using with large scale and on-line/real-time applications. Other well-developed methods (but less standard) are immunological or nucleic acid-based methods, such as enzyme-linked immunosorbent assay (ELISA) (Macedo-Silva et al. 2000), and methods that utilize polymerase chain reaction (PCR). The methods are not rapid, destructive and require time, personnel and laboratories for experiments and not real-time solutions.

The obvious opportunities to improve the process with state-of-the-art software applications, frameworks and image detection mechanisms with applying AI/machine learning and data analytics. Not only the python language frameworks for imaging processing and recognition, the developed and mature frameworks for object detection in deep learning and computer vision which have been used in face, vehicle detection or in pedestrian counting and self-driving car, etc. The following items are the common approaches for object detection: ImageAI, Single Shot Detectors, YOLO (You only look once), Region-based Convolutional Neural Networks (R-CNNs). YOLO family has three members YOLO (2015), YOLO9000 (2016) and YOLOv3 (2018) and R-CNN family has 4 members R-CNN (2014), Fast R-CNN (2015), Faster R-CNN (2016) and Mask R-CNN (2017). The latter version has solved some limitations of the former version.

Besides the significant increasing software applications and frameworks, the big enhancement in hardware for the variant type of smart sensors or biosensors, for the better imaging system has brought the huge advantages in developing the expected solutions. A smartphone-based biosensor was developed to detect and quantify microbial contamination (mainly for E-coli) on ground beef, consisting of an 880 nm NIR LED and a smartphone (utilizing its digital camera, software application, and an internal gyro sensor) (Pei-Shih Liang, Tu San Park & Jeong-Yeol Yoon. 2014). Furthermore, combinations of the effectiveness of hyperspectral imaging and machine learning techniques (classification and prediction) have been proved that it is a non destructive method to detect, classify, and quantify plant- and animal-based adulterants in minced beef and pork. (Admed Rady et al. 2020). 

## Brief of what team has done
From the literature review result, team has decided to make research about the follow topics in order and each members were assigned to work on some parts of the list.
-	Learning about camera types which mainly used in this industry, four camera types (RBG, IR, Multispectral and Hyperspectral) have been analysed, compared the selected the most adapted type for the solution.
-	Besides the camera, the shutter was also investigated to improve the quality of the captured image. It would help to improve the accuracy of processing images.
-	Designing the layout for inspection room that using the rolling rollers
-	Analyse development and deployment model with docker container
-	4 main types of segmentation methodologies were analyzed and selected. UNET was selected for semantic segmentation and Mask RCNN was selected for instance segmentation.
-	Preparing the data of citispace dataset then training the model the detecting the images with both UNET and MaskRCNN
-	Tuning the model to have better accuracy values.

## Brief review of what I have done
-	As a part of learning about the camera type, IR cameras are analyzed to see if they are applicable for the solution. 
-	At the beginning of working on the code example of MaskRCNN and UNET, team got problems with different operating systems (some use Ms Windows, some use MacOS). In collaborating with other members, build the docker container that allows us to work in a similar environment together.
-	MaskRCNN was chosen to develop the MVP with citispaces dataset, Annie and I have focused in training and testing the model from scratch based on provided images and JSON files in the cityspace dataset. Once the model was built, tested, we did tuning the model to have better AP values. 
 
# Design Concept
## High level design
![High Level Design](/images/1.jpg)

In this overview, there are some important components for the system: 
First of all is the rolling roller and the rail system, because of the very high cost of a hyperspectral camera, so the special design of the rail system in the inspection room (like a circle) can help us to save the cost of buying more than 1 hyperspectral cameras for multiple angle capturing the the beef. With the round rail nearly like a cicle, multiple shots of the beef at different timestamp as the example beef#2 in the picture, it will be captured in 5 times (t1, t2, t3, t4, t5) during the journey of moving on the rounding rail.

The second component of the system obviously the hyperspectral camera which will help the system to take good enough quality images of the beef. The camera is supporting by some different special leds to supplement the different lights in order help the captured images has been captured in enough light ranges (in nm).

The videos or images will be streaming from the hyperspectral camera to the Analytic hub which will sample the stream to a collection of the examine beefs then the images are moved to prediction engine has been build based on UNET or MaskRCNN (deployed in the Analytic hub). The analytic results will be collected and showed to the observers on his screen. The nearly realtime result will help observer to take equivalent actions on the examining beef.  

## Hardware selection
### Camera Selection
Contaminants can be quite small, and vary in colour and size, creating challenges in detection. So, to facilitate efficient image processing and segmentation, we require high resolution images of carcasses the maximise the contrast between meat and contaminants.

Four type of camera are chosen to analyze further including RGB, Infrared, Multispectral and Hyperspectral. RGB is much like the human eye that just focusing on the sensitive with red, green and blue light. Teams have had success using RGB cameras to identify contaminants with moderate success (Adi et al. 2017 reports 84% accuracy for testing). RGB cameras have their limitation in wavelengths of the light (range 360 nm – 830 nm). There are other wavelengths of light that can be detected by infrared or multispectral and hyperspectral imaging systems. Multiple studies report greatest success with hyperspectral camera in detecting the small things in food so it will be discussed in detail.

A hyperspectral imaging systems obtain spectrum data within and beyond the human eye, including visible spectrum (300-800 nm), visible and a near-infrared (400-1000 nm), shortwave near infrared (900-1700 nm) and long-wave near-infrared (1000-2500 nm). As detailed by Xiong et al. (2015) after significant processing, hyperspectral images have been proven to maximise the contrast between xero-tolerance contaminants and meat tissues in chicken carcasses. 

Figure 2 from Park et al. (2006) give a striking demonstration of the effectiveness of these solutions, and attests to the ability of hyperspectral image capture and processing to detect and contrast contaminants from the rest of the poultry carcass. This is most strongly highlighted in the final (e) panel.

Figure 02. Hyperspectral images of poultry carcass
![Figure 02. Hyperspectral images of poultry carcass](/images/2.jpg)

## Shutter Selection
After researching 4 types of camera, the hyperspectral camera is recommended. However, it is the most expensive camera, costing upwards of $50,000. To to eliminate the cost, a single camera will be used in the MVP solution. Once the solution shows that it is effectively and efficiency in detecting the contaminants, the solution will consider a bigger number of hyperspectral camera when deploying to the industry.

Also, to utilize the camera to capture different angles of the beef, a special shape of rail was designed in the inspection room to rotate the carcase continuously and so over the time rolling in the round rail, the carcase was imaged multiple times. And the design also allow a constant movement of the carcass to improve the productivity of the whole process. 

When consider two different types (rolling and global) of the shutter using for hyperspectral, we see that the  result of global shutter oviously better than the output of using a rolling shutter. So, the global shutter is preferred to use in this case. As the demonstrated image below, using a global shutter will also cut down on image processing time as there will be no need for an algorithm to flatten out the spatial distortion.

Figure 03 – Rolling Shutter vs Global Shutter
![Figure 03 – Rolling Shutter vs Global Shutter](/images/3.jpg)

## Software
### Data / Image Preparation
After going over the rounded track, the carcases have been imaged from multiple angles (nearly 3600). The images or streamings are forwarding to the analytic hub for storing then processing. As the special characteristic of hyperspectral image which made up of hundreds of continuous wavebands for each angle position of the studied sample. It means that each pixel in a hyperspectral image contains the spectrum of the specific area in the carcase. 

And after combining the similar images, removing noise in the pre-processing process, the image will be extracted the further information and then highlight the variation between multiple regions of the image. For example, textural analysis by grey level co-occurrence matrix and Gabor transform are two widely used algorithms of hyperspectral image processing (Elmasry, Barbin, Sun & Allen (2012).

After image processing, to establish a corresponding discrimination model, preparing for model training step, some dimensional reduction methods are applied including principal component analysis, independent component analysis, and genetic algorithm.

### Classification & Segmentation 
A number of segmentation solutions have been investigated and demonstrated the effective identification of contaminants which captured by multispectral imaging system. For instance, Windham et al. (2005), Wu et al. (2017) and Elmasry, Barbin, Sun & Allen (2012) have illutrated the accuracy in the detection of existing contaminants in poultry carcasses with minor false values.

However, these solutions were using the linear statistic approach to detect the result based on the relationship of some main features of date. This will stop the continuous improvement of the model during running in the production. So, some other more sophisticated segmentations should be explored to keep the model learning over the running time (more attributes of ML should be used in this case) without requiring the linear data. 

As summary below, there are four methodologies are researching. 
**Image Classification & Object Localisation/Detection**
As shown, image classification and object localisation or object detection are seemly not suitable for the requirement of this project which needs to show exactly the areas in that the carcass affected by one or some of contaminants.

Figure 04. Different results of different types 
![Different results of different types](/images/4.jpg)

Two other considering methods semantic or instance segmentation provide their advantages in showing not only the areas (in pixel and with specific color) but also the labels for the objects.

**Semantic segmentations** 
In this model, it will assign an object category label to each pixel in the image. As in the example picture in above figure, the sheep pixels are coloured orange, road pixels are coloured brown, and grass pixels are green. However, all the sheep in the image are coloring with the same label and in the same area.

**Instance segmentations**
In this model, it will performs to describe above not only for semantic segmentation, but also assigns for any object a label to each pixel in the image. Revisit the example, the pixels for each individual sheep are labelled separately with different colors and separated labels. Instead of having a generic “sheep” pixel class, now there are three classes for the 3 sheep shown: sheep1, sheep2, and sheep3.

# UNET & Mask RCNN Segementation Implementation
Both semantic with U-NET and instance segmentation with MaskRCNN methods are used, and then determine the most appropriate result for each method. 

U-NET can provide the result faster than Mask R-CNN while the Mask R-CNN can achieve better accuracy. The comparison in table below      for some key attributes. 

Table 01. Comparison between U-NET and Mask R-CNN
![Comparison between U-NET and Mask R-CNN](/images/5.jpg)



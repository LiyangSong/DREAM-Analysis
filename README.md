# DREAM Dataset Analysis

## Background

The DREAM dataset originates from the [DREAM program](https://dream2020.github.io/DREAM/) - Development of Robot-Enhanced therapy for children with Autism spectrum disorders. This large-scale European research program aims to advance research into Autism Spectrum Disorder (ASD), one of the most common childhood developmental disorders, and to develop innovative methods and tools to support the diagnosis and therapy of children with ASD. 

The dataset contains behavioral data from over 3,000 therapy sessions involving 61 children diagnosed with ASD, aged between 3 and 6 years. These sessions include an initial diagnosis, six therapeutic interventions, and a final diagnosis. Among the 61 children, 41 completed the entire therapy process, while the remaining participants completed an average of 4.5 interventions. Each session was recorded using three RGB cameras and two RGBD (Kinect) cameras, providing detailed information on the children’s behavior during therapy.

This dataset is particularly valuable for studying child-robot interactions, focusing on how children with ASD respond differently to robots compared to human partners. By analyzing these behaviors, researchers can fine-tune robotic interventions for enhanced therapeutic outcomes.

![image-20241015203028722.png](/img/image-20241015203028722.png)

![image-20241119192331336.png](/img/image-20241119192331336.png)

![image-20241119191305598.png](/img/image-20241119191305598.png)

## Data Structure

All data in the dataset is stored in JSON format, with a detailed structure presented below. 

```json
{
    "$id": "User 37_18_Intervention 2_20171102_123242.369000.json",
    "$schema": "https://raw.githubusercontent.com/dream2020/data/master/specification/dream.1.1.json",
    "ados": {
        "preTest": {
            "communication": 2,
            "interaction": 5,
            "module": 1.0,
            "play": 1,
            "protocol": "ADOS-G",
            "socialCommunicationQuestionnaire": 23,
            "stereotype": 0,
            "total": 7
        }
    },
    "condition": "RET",
    "date": "2017-11-02T12:32:42.369000",
    "eye_gaze": {"rx": [],"ry": [],"rz": []},
    "frame_rate": 25.1,
    "head_gaze": {"rx": [],"ry": [],"rz": []},
    "participant": {"ageInMonths": 47,"gender": "male","id": 37},
    "skeleton": {
        "elbow_left": {"confidence": [],"x": [],"y": [],"z": []},
        "elbow_right": {"confidence": [],"x": [],"y": [],"z": []},
        "hand_left": {"confidence": [],"x": [],"y": [],"z": []},
        "hand_right": {"confidence": [],"x": [],"y": [],"z": []},
        "head": {"confidence": [],"x": [],"y": [],"z": []},
        "sholder_center": {"confidence": [],"x": [],"y": [],"z": []},
        "sholder_left": {"confidence": [],"x": [],"y": [],"z": []},
        "sholder_right": {"confidence": [],"x": [],"y": [],"z": []},
        "wrist_left": {"confidence": [],"x": [],"y": [],"z": []},
        "wrist_right": {"confidence": [],"x": [],"y": [],"z": []}
    },
    "task": "TT"
}
```

## How to Reproduce

1. Set up the environment.

```bash
conda env create -f environment.yml
conda activate dream-analysis-env
```

2. Download the raw dataset using `Make` command.

```bash
make download
```

3. Execute symmetry analysis and output a csv format result in `results/analysis` folder.

```bash
make analysis
```

4. Execute skeleton visualization and output mp4 format videos in `results/visualization` folder.

```bash
make visualization
```

## RET Framework

The dataset includes variables related to specific tasks, such as **imitation**, **joint attention**, and **turn-taking**. These tasks assess the child's ability to mimic the robot’s actions, follow visual cues, and interact reciprocally in structured scenarios.

##### Imitation

This task evaluates how accurately a child can mimic the robot's movements, emotional expressions, and sounds. Interactive imitation games are conducted to teach motor behaviors and improve imitation skills. The robot provides verbal instructions for the child to replicate its actions, with the imitation tasks organized by various levels of difficulty:
- **Level 1 (Imitation with Objects)**: The child imitates actions involving objects, such as moving a toy car or pretending to drink from a cup.
- **Level 2 (Imitation of Gestures)**: The child imitates meaningful movements, such as waving and saying "bye-bye."
- **Level 3 (Imitation of Non-meaningful Movements)**: The child imitates abstract movements without an inherent meaning.

##### Joint Attention

This task captures behaviors such as gaze direction, gestures, and verbalizations aimed at directing shared attention to specific objects.

The task begins with a brief explanation provided to the child, such as: “Now we will play another game. In this game, I will show you the objects I’ve seen in an office.” Two pictures are then displayed simultaneously on a large touch-screen table, one on the left and one on the right. The child must look at the picture indicated by the robot. The complexity of the task increases across three levels:
- **Level 1**: The robot uses a combination of verbal action (e.g., “Look!”), gestural action (e.g., pointing to a picture), and gaze action (e.g., making eye contact and then shifting its gaze to the picture).
- **Level 2**: The robot omits verbal cues and only uses gestural and gaze actions to direct the child’s attention.
- **Level 3**: The robot relies solely on gaze cues, omitting both verbal and gestural actions.

##### Turn-taking

This task involves tracking the child’s responses, including contingent utterances, emotional expressions, and rational or maladaptive behaviors during interactions with the robot. The child and robot take turns in a series of structured activities presented on a touch-screen tablet (Sandtray). These activities include sharing information, categorizing items, and continuing repeating patterns. Each sub-task begins with an instruction or question from the robot, followed by the child’s response and the robot’s feedback based on the behavior.

**1. Sharing information**: 
  - Five pictures are displayed on the tablet. The child selects a picture during their turn and waits while the robot selects a picture during its turn.

**2. Categories**:
  - **Level 1**: Three pictures are displayed (two representing categories and one item to be categorized). The child categorizes the item during their turn and waits while the robot categorizes during its turn. The categories are simple and age-appropriate (e.g., fruits vs. vegetables), and items appear one by one.
  - **Level 2**: Ten pictures are displayed (two categories and eight items to be categorized). The child categorizes one item at a time during their turn and waits for the robot to do the same. The categories are more complex (e.g., ground vehicles vs. water vehicles).

**3. Patterns**:
  - **Level 1**: Six pictures are displayed, including a repetitive pattern of two or three items. The child continues the pattern during their turn and waits for the robot to add to the pattern during its turn. The task focuses on simple criteria like geometric shapes (e.g., rectangle, rectangle, triangle, rectangle, ...).
  - **Level 2**: Ten pictures are displayed, with a more complex pattern consisting of four repeating items. The child categorizes based on two criteria—geometric shape and color (e.g., green square, star, orange square, circle, ...).

## Resources

- [The DREAM Dataset: Supporting a data-driven study of autism spectrum disorder and robot enhanced therapy](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0236939#abstract0)
- [DREAM Dataset](https://github.com/dream2020/data)
- [Open Source DREAM Project](https://dream2020.github.io/DREAM/)
- [DREAM_Deliverable_D1.1 Intervention Definition](https://dream2020.github.io/DREAM/deliverables/DREAM_Deliverable_D1.1.pdf)
- [DREAM_Deliverable_D1.4 Manual of best practice in robot enhanced therapy for
  autism spectrum disorder](https://dream2020.github.io/DREAM/deliverables/D1.4-Manual-of-best-practice-in-RET-10.05.pdf)
- [DREAM_Deliverable_D4.1 Sensorized Therapy Room Design and
  Algorithms for Data Sensing and Interpretation](https://dream2020.github.io/DREAM/deliverables/DREAM_Deliverable_D4.1.pdf)
- [DREAM Data Visualizer](https://github.com/dream2020/DREAM-data-visualizer)
- [Sensing-enhanced Therapy System for Assessing
  Children with Autism Spectrum Disorders: A
  Feasibility Study](https://ieeexplore.ieee.org/document/8502783)
- [Computer Vision to the Rescue: Infant Postural Symmetry Estimation from
  Incongruent Annotations](https://arxiv.org/pdf/2207.09352)
- [Two-Eye Model-Based Gaze Estimation from A Kinect Sensor](https://ieeexplore.ieee.org/document/7989194)



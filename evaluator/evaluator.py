from pydantic import BaseModel, Field
from utils.visualize_layout_bottom import create_grid

class SceneEvaluationModel(BaseModel):
    physical_reasoning: str = Field(description="The reasoning for physical reasonability")
    semantic_reasoning: str = Field(description="The reasoning for semantic reasonability")
    layout_reasoning: str = Field(description="The reasoning for layout reasonability")
    instruction_alignment_reasoning: str = Field(description="The reasoning for instruction alignment")
    
    physical_score: int = Field(description="Score for physical reasonability (0-4)", ge=0, le=4)
    semantic_score: int = Field(description="Score for semantic reasonability (0-4)", ge=0, le=4)
    layout_score: int = Field(description="Score for layout reasonability (0-4)", ge=0, le=4)
    instruction_alignment_score: int = Field(description="Score for instruction alignment (0-4)", ge=0, le=4)

def fmt(image_np, instruction):
    ret = {}
    ret['user_prompt'] = \
f"""
[Role]
You are a professional scene evaluator with expertise in interior design, physics, and spatial relationships.

[Task]
Your task is to evaluate the quality of a 3D scene from its top-down view image. The position and size of each object is represented by a colored rectangle with a label.

[Input]
Input Image: This image presents a top-down view of a scene with various objects and furniture.
Instruction: {instruction}

[IMPORTANT INSTRUCTIONS FOR UNDERSTANDING THE IMAGE]
In the image:
- Colored rectangles represent the position and size of each object
- Text labels above each rectangle indicate the object type
- Arrows on the rectangles show the orientation/facing direction of the objects

[IMPORTANT INSTRUCTIONS FOR OVERLAP DETECTION]
In the image, objects are represented by colored rectangles with labels. When rectangles visually overlap or share the same space, this indicates physical overlapping of furniture, which is a serious problem:

- When one colored rectangle visually intersects or overlaps with another colored rectangle, those objects are physically overlapping
- Text labels may sometimes appear to overlap with other rectangles - this is NOT what you should focus on
- Focus on the colored rectangle boundaries themselves - when these overlap, it means the objects would be occupying the same physical space
- CAREFULLY EXAMINE the entire image to spot ALL overlaps between colored rectangles

[Basic Requirements]
Follow these steps carefully to ensure the task is completed with clarity and accuracy:
1. **Understand the Context**: Based on the input image and the original instruction, first assess the scene as a whole and describe what you see.
2. **Examine Every Object Pair**: For EACH pair of objects, explicitly check if they are physically overlapping or intersecting.
3. **Step-by-step Reasoning**: For each evaluation criterion, provide detailed reasoning BEFORE assigning any scores.

[Evaluation Criteria]
You need to evaluate the scene quality on FORE aspects, each scored from 0 to 4 (higher is better):

1. **Physical Reasonability (0-4)**:
   - CRITICAL: Check every pair of objects to identify if their colored rectangles overlap or intersect. If one object is placed above on another on the top-down view, it is considered as overlapping.
   - Score 0: If multiple objects have significant overlapping (objects sharing >30% area)
   - Score 1: If objects have significant overlapping (one pair sharing >30% area)
   - Score 2: If objects have moderate overlapping (10-30% area)
   - Score 3: If objects have minor overlapping (<10% area)
   - Score 4: If no objects' colored rectangles are overlapping at all

2. **Semantic Reasonability (0-4)**:
   - Evaluate whether objects are arranged according to their functional purposes.
   - Score 0: Objects are placed in ways that make no functional sense
   - Score 1: Objects are placed with very limited functional consideration
   - Score 2: Objects are placed with some functional consideration
   - Score 3: Objects are mostly arranged according to their functional purposes
   - Score 4: Objects are perfectly arranged according to their functional purposes

3. **Layout Reasonability (0-4)**:
   - Assess the spatial relationships between objects.
   - Check if there's appropriate spacing for movement and use.
   - Evaluate if the overall arrangement follows common design principles.
   - IMPORTANT: Evaluate if the orientation (arrow on the rectangle) of objects is reasonable:
     * Furniture should face appropriate directions (e.g., sofas facing TV/coffee tables)
     * Chairs should face tables or conversation areas
     * Beds should typically be against walls with headboards
     * Desks should face away from walls or toward windows
     * Storage furniture should be accessible (doors/drawers can open)
     * etc.
   - Score 0: Very poor layout with significant spacing problems and incorrect orientations
   - Score 1: Poor layout with inadequate spacing and mostly incorrect object orientations
   - Score 2: Basic layout with some consideration for spacing and partially correct orientations
   - Score 3: Good layout with appropriate spacing and mostly correct object orientations
   - Score 4: Excellent layout with optimal spacing, adherence to design principles, and perfect object orientations

4. **Instruction Alignment (0-4)**:
   - Evaluate whether the scene meets the requirements specified in the original instruction or prompt. This includes:
     * Whether all required objects are present
     * Whether specified relationships between objects are satisfied
   - IMPORTANT: Do not consider the style of the scene, only consider the position and orientation requirements in the instruction.
   - Score 0: The scene does not meet any of the instruction requirements
   - Score 1: The scene meets very few instruction requirements
   - Score 2: The scene meets some instruction requirements
   - Score 3: The scene meets most instruction requirements
   - Score 4: The scene fully meets all instruction requirements

[Process]
1. First, provide detailed reasoning for Physical Reasonability:
   - List every pair of objects and explicitly state whether they are physically overlapping or intersecting
   - Make your final physical reasoning conclusion based on these observations

2. Next, provide detailed reasoning for Semantic Reasonability - analyze functional relationships, contextual appropriateness, and whether the objects match the room type.

3. Then, provide detailed reasoning for Layout Reasonability - evaluate spatial relationships, design principles, and object orientations:
   - For each major furniture piece, comment on whether its orientation is appropriate
   - Consider how the orientations affect usability and interaction between objects
   - Evaluate if orientations follow conventional interior design practices

4. Next, provide detailed reasoning for Instruction Alignment - analyze whether the scene meets position and orientation requirements in the original instruction or prompt.

5. ONLY AFTER completing all reasoning steps, assign scores based on your analysis.

Remember: Your most important task is to correctly identify when objects are physically overlapping or intersecting in the image, as this indicates objects occupying the same physical space, which is unrealistic.
"""

    ret['check_fn'] = lambda _: None  # No need for check function as we're using Pydantic
    ret['image_np'] = image_np[:,:,::-1]
    return ret


def textual_fmt(text_layout, instruction):
    ret = {}
    ret['user_prompt'] = \
f"""
[Role]
You are a professional scene evaluator with expertise in interior design, physics, and spatial relationships.

[Task]
Your task is to evaluate the quality of a 3D scene from its textual layout. The position and size of each object is represented by a colored rectangle with a label.

[Input]
Textual Layout: {text_layout}
Instruction: {instruction}

[Basic Requirements]
Follow these steps carefully to ensure the task is completed with clarity and accuracy:
1. **Understand the Context**: Based on the input textual layout and the original instruction, first assess the scene as a whole and describe what you see.
2. **Examine Every Object Pair**: For EACH pair of objects, explicitly check if they are physically overlapping or intersecting.
3. **Step-by-step Reasoning**: For each evaluation criterion, provide detailed reasoning BEFORE assigning any scores.

[Evaluation Criteria]
You need to evaluate the scene quality on FORE aspects, each scored from 0 to 4 (higher is better):

1. **Physical Reasonability (0-4)**:
   - CRITICAL: Check every pair of objects to identify if their colored rectangles overlap or intersect.
   - Score 0: If multiple objects have significant overlapping (objects sharing >30% area)
   - Score 1: If objects have significant overlapping (one pair sharing >30% area)
   - Score 2: If objects have moderate overlapping (10-30% area)
   - Score 3: If objects have minor overlapping (<10% area)
   - Score 4: If no objects' colored rectangles are overlapping at all

2. **Semantic Reasonability (0-4)**:
   - Evaluate whether objects are arranged according to their functional purposes.
   - Score 0: Objects are placed in ways that make no functional sense
   - Score 1: Objects are placed with very limited functional consideration
   - Score 2: Objects are placed with some functional consideration
   - Score 3: Objects are mostly arranged according to their functional purposes
   - Score 4: Objects are perfectly arranged according to their functional purposes

3. **Layout Reasonability (0-4)**:
   - Assess the spatial relationships between objects.
   - Check if there's appropriate spacing for movement and use.
   - Evaluate if the overall arrangement follows common design principles.
   - IMPORTANT: Evaluate if the orientation (arrow on the rectangle) of objects is reasonable:
     * Furniture should face appropriate directions (e.g., sofas facing TV/coffee tables)
     * Chairs should face tables or conversation areas
     * Beds should typically be against walls with headboards
     * Desks should face away from walls or toward windows
     * Storage furniture should be accessible (doors/drawers can open)
     * etc.
   - Score 0: Very poor layout with significant spacing problems and incorrect orientations
   - Score 1: Poor layout with inadequate spacing and mostly incorrect object orientations
   - Score 2: Basic layout with some consideration for spacing and partially correct orientations
   - Score 3: Good layout with appropriate spacing and mostly correct object orientations
   - Score 4: Excellent layout with optimal spacing, adherence to design principles, and perfect object orientations

4. **Instruction Alignment (0-4)**:
   - Evaluate whether the scene meets the requirements specified in the original instruction or prompt. This includes:
     * Whether all required objects are present
     * Whether specified relationships between objects are satisfied
   - IMPORTANT: Do not consider the style of the scene, only consider the position and orientation requirements in the instruction.
   - Score 0: The scene does not meet any of the instruction requirements
   - Score 1: The scene meets very few instruction requirements
   - Score 2: The scene meets some instruction requirements
   - Score 3: The scene meets most instruction requirements
   - Score 4: The scene fully meets all instruction requirements

[Process]
1. First, provide detailed reasoning for Physical Reasonability:
   - List every pair of objects and explicitly state whether they are physically overlapping or intersecting
   - Make your final physical reasoning conclusion based on these observations

2. Next, provide detailed reasoning for Semantic Reasonability - analyze functional relationships, contextual appropriateness, and whether the objects match the room type.

3. Then, provide detailed reasoning for Layout Reasonability - evaluate spatial relationships, design principles, and object orientations:
   - For each major furniture piece, comment on whether its orientation is appropriate
   - Consider how the orientations affect usability and interaction between objects
   - Evaluate if orientations follow conventional interior design practices

4. Next, provide detailed reasoning for Instruction Alignment - analyze whether the scene meets position and orientation requirements in the original instruction or prompt.

5. ONLY AFTER completing all reasoning steps, assign scores based on your analysis.

Remember: Your most important task is to correctly identify when objects are physically overlapping or intersecting in the image, as this indicates objects occupying the same physical space, which is unrealistic.
"""

    ret['check_fn'] = lambda _: None  # No need for check function as we're using Pydantic
    ret['image_np'] = None
    return ret


def prm_fmt(image_np, instruction):
    ret = {}
    ret['user_prompt'] = \
f"""
[Role]
You are a professional scene evaluator with expertise in interior design, physics, and spatial relationships. As a Process Reward Model (PRM), your goal is to evaluate intermediate states of scene arrangements to guide the scene generation process.

[Task]
Your task is to evaluate the quality and potential of a 3D scene's current state from its top-down view image. You should focus on both the current arrangement quality and its potential for improvement. The position and size of each object is represented by a colored rectangle with a label.

[Input]
Input Image: This image presents a top-down view of a scene with various objects and furniture.
Instruction: {instruction}

[IMPORTANT INSTRUCTIONS FOR UNDERSTANDING THE IMAGE]
In the image:
- Colored rectangles represent the position and size of each object
- Text labels above each rectangle indicate the object type
- Arrows on the rectangles show the orientation/facing direction of the objects

[IMPORTANT INSTRUCTIONS FOR OVERLAP DETECTION]
In the image, objects are represented by colored rectangles with labels. When rectangles visually overlap or share the same space, this indicates physical overlapping of furniture, which is a serious problem:

- When one colored rectangle visually intersects or overlaps with another colored rectangle, those objects are physically overlapping
- Text labels may sometimes appear to overlap with other rectangles - this is NOT what you should focus on
- Focus on the colored rectangle boundaries themselves - when these overlap, it means the objects would be occupying the same physical space
- CAREFULLY EXAMINE the entire image to spot ALL overlaps between colored rectangles

[Basic Requirements]
Follow these steps carefully to ensure the task is completed with clarity and accuracy:
1. **Understand the Context**: Based on the input image and the original instruction, first assess the scene as a whole and describe what you see.
2. **Examine Every Object Pair**: For EACH pair of objects, explicitly check if they are physically overlapping or intersecting.
3. **Step-by-step Reasoning**: For each evaluation criterion, provide detailed reasoning BEFORE assigning any scores.

[Evaluation Criteria]
You need to evaluate the scene quality on FOUR aspects, each scored from 0 to 4 (higher is better). Remember that you are evaluating an intermediate state, so consider both current quality and future potential:

1. **Physical Reasonability (0-4)**:
   - CRITICAL: Check every pair of objects to identify if their colored rectangles overlap or intersect. If one object is placed above on another on the top-down view, it is considered as overlapping.
   - Score 0: If multiple objects have significant overlapping (objects sharing >30% area)
   - Score 1: If objects have significant overlapping (one pair sharing >30% area)
   - Score 2: If objects have moderate overlapping (10-30% area)
   - Score 3: If objects have minor overlapping (<10% area)
   - Score 4: If no objects' colored rectangles are overlapping at all

2. **Semantic Reasonability (0-4)**:
   - Evaluate whether objects are arranged according to their functional purposes and whether current arrangements show potential for good functional relationships
   - Score 0: Objects are placed in ways that make no functional sense and show no potential
   - Score 1: Objects show limited functional consideration but have room for improvement
   - Score 2: Objects show some functional consideration with clear potential for enhancement
   - Score 3: Objects mostly show good functional relationships with minor improvements possible
   - Score 4: Objects are perfectly arranged according to their functional purposes

3. **Layout Reasonability (0-4)**:
   - Assess both current spatial relationships and potential for improvement
   - Consider whether current spacing and orientations can be enhanced in future steps
   - IMPORTANT: Evaluate if the orientation (arrow on the rectangle) of objects is reasonable or shows potential for improvement:
     * Furniture should face or show potential to face appropriate directions
     * Consider whether current positions allow for proper orientation adjustments
     * Evaluate if there's sufficient space for proper orientation changes
   - Score 0: Very poor layout with limited potential for improvement
   - Score 1: Poor layout but shows some potential for enhancement
   - Score 2: Basic layout with clear opportunities for improvement
   - Score 3: Good layout with potential for excellence
   - Score 4: Excellent layout with optimal spacing and orientations

4. **Instruction Alignment (0-4)**:
   - As a Process Reward Model (PRM), evaluate whether the current scene arrangement shows potential progress towards meeting the instruction requirements:
     * Whether the required objects that are already placed show promising arrangements
     * Whether the current object positions and orientations indicate a good direction towards satisfying the final requirements
   - IMPORTANT: Focus on evaluating the potential of the current intermediate state, not whether it fully meets all requirements yet
   - Score 0: The current arrangement shows no potential for meeting the instruction requirements
   - Score 1: The current arrangement shows limited potential, with most placed objects poorly positioned
   - Score 2: The current arrangement shows some potential, with some objects properly positioned or oriented
   - Score 3: The current arrangement shows good potential, with most placed objects showing promising positions/orientations
   - Score 4: The current arrangement shows excellent potential for satisfying all instruction requirements

[Process]
1. First, provide detailed reasoning for Physical Reasonability:
   - List every pair of objects and explicitly state whether they are physically overlapping or intersecting
   - Make your final physical reasoning conclusion based on these observations

2. Next, provide detailed reasoning for Semantic Reasonability - analyze functional relationships, contextual appropriateness, and whether the objects match the room type.

3. Then, provide detailed reasoning for Layout Reasonability - evaluate spatial relationships, design principles, and object orientations:
   - For each major furniture piece, comment on whether its orientation is appropriate
   - Consider how the orientations affect usability and interaction between objects
   - Evaluate if orientations follow conventional interior design practices

4. Next, provide detailed reasoning for Instruction Alignment - analyze whether the scene meets position and orientation requirements in the original instruction or prompt.

5. ONLY AFTER completing all reasoning steps, assign scores based on your analysis.

Remember: Your most important task is to correctly identify when objects are physically overlapping or intersecting in the image, as this indicates objects occupying the same physical space, which is unrealistic.
"""

    ret['check_fn'] = lambda _: None  # No need for check function as we're using Pydantic
    ret['image_np'] = image_np[:,:,::-1]
    return ret


def textual_prm_fmt(textual_layout, instruction):
    ret = {}
    ret['user_prompt'] = \
f"""
[Role]
You are a professional scene evaluator with expertise in interior design, physics, and spatial relationships. As a Process Reward Model (PRM), your goal is to evaluate intermediate states of scene arrangements to guide the scene generation process.

[Task]
Your task is to evaluate the quality and potential of a 3D scene's current state from its textual layout. You should focus on both the current arrangement quality and its potential for improvement. The position and size of each object is represented by a colored rectangle with a label.

[Input]
Textual Layout: {textual_layout}
Instruction: {instruction}

[Basic Requirements]
Follow these steps carefully to ensure the task is completed with clarity and accuracy:
1. **Understand the Context**: Based on the input textual layout and the original instruction, first assess the scene as a whole and describe what you see.
2. **Examine Every Object Pair**: For EACH pair of objects, explicitly check if they are physically overlapping or intersecting.
3. **Step-by-step Reasoning**: For each evaluation criterion, provide detailed reasoning BEFORE assigning any scores.

[Evaluation Criteria]
You need to evaluate the scene quality on FOUR aspects, each scored from 0 to 4 (higher is better). Remember that you are evaluating an intermediate state, so consider both current quality and future potential:

1. **Physical Reasonability (0-4)**:
   - CRITICAL: Check every pair of objects to identify if their colored rectangles overlap or intersect.
   - Score 0: If multiple objects have significant overlapping (objects sharing >30% area)
   - Score 1: If objects have significant overlapping (one pair sharing >30% area)
   - Score 2: If objects have moderate overlapping (10-30% area)
   - Score 3: If objects have minor overlapping (<10% area)
   - Score 4: If no objects' colored rectangles are overlapping at all

2. **Semantic Reasonability (0-4)**:
   - Evaluate whether objects are arranged according to their functional purposes and whether current arrangements show potential for good functional relationships
   - Score 0: Objects are placed in ways that make no functional sense and show no potential
   - Score 1: Objects show limited functional consideration but have room for improvement
   - Score 2: Objects show some functional consideration with clear potential for enhancement
   - Score 3: Objects mostly show good functional relationships with minor improvements possible
   - Score 4: Objects are perfectly arranged according to their functional purposes

3. **Layout Reasonability (0-4)**:
   - Assess both current spatial relationships and potential for improvement
   - Consider whether current spacing and orientations can be enhanced in future steps
   - IMPORTANT: Evaluate if the orientation (arrow on the rectangle) of objects is reasonable or shows potential for improvement:
     * Furniture should face or show potential to face appropriate directions
     * Consider whether current positions allow for proper orientation adjustments
     * Evaluate if there's sufficient space for proper orientation changes
   - Score 0: Very poor layout with limited potential for improvement
   - Score 1: Poor layout but shows some potential for enhancement
   - Score 2: Basic layout with clear opportunities for improvement
   - Score 3: Good layout with potential for excellence
   - Score 4: Excellent layout with optimal spacing and orientations

4. **Instruction Alignment (0-4)**:
   - As a Process Reward Model (PRM), evaluate whether the current scene arrangement shows potential progress towards meeting the instruction requirements:
     * Whether the required objects that are already placed show promising arrangements
     * Whether the current object positions and orientations indicate a good direction towards satisfying the final requirements
   - IMPORTANT: Focus on evaluating the potential of the current intermediate state, not whether it fully meets all requirements yet
   - Score 0: The current arrangement shows no potential for meeting the instruction requirements
   - Score 1: The current arrangement shows limited potential, with most placed objects poorly positioned
   - Score 2: The current arrangement shows some potential, with some objects properly positioned or oriented
   - Score 3: The current arrangement shows good potential, with most placed objects showing promising positions/orientations
   - Score 4: The current arrangement shows excellent potential for satisfying all instruction requirements

[Process]
1. First, provide detailed reasoning for Physical Reasonability:
   - List every pair of objects and explicitly state whether they are physically overlapping or intersecting.
   - Make your final physical reasoning conclusion based on these observations

2. Next, provide detailed reasoning for Semantic Reasonability - analyze functional relationships, contextual appropriateness, and whether the objects match the room type.

3. Then, provide detailed reasoning for Layout Reasonability - evaluate spatial relationships, design principles, and object orientations:
   - For each major furniture piece, comment on whether its orientation is appropriate
   - Consider how the orientations affect usability and interaction between objects
   - Evaluate if orientations follow conventional interior design practices

4. Next, provide detailed reasoning for Instruction Alignment - analyze whether the scene meets position and orientation requirements in the original instruction or prompt.

5. ONLY AFTER completing all reasoning steps, assign scores based on your analysis.

Remember: Your most important task is to correctly identify when objects are physically overlapping or intersecting in the textual layout, as this indicates objects occupying the same physical space, which is unrealistic.
"""

    ret['check_fn'] = lambda _: None  # No need for check function as we're using Pydantic
    ret['image_np'] = None
    return ret
    

def mcts_evaluator(node, resolution, render_size, instruction, get_response_task, use_image=True, prm=False):
    """
    Evaluate a scene based on its top-down view.
    
    Args:
        image_np: Numpy array of the scene image (BGR format)
        get_response_task: Function to get response from LLM
        
    Returns:
        Dict containing the evaluation results
    """
    vis_furnitures_list = []
    named_colors = [
      'lightcoral', 'salmon', 'sienna', 'darkorange', 'gold', 'olive', 'yellow', 'yellowgreen', 'green', 'lightseagreen', 'deepskyblue', 'dodgerblue', 'blue', 'darkblue', 'indigo', 'purple', 'magenta', 'hotpink', 'pink', 'lightpink', 'white', 'lightgrey'
    ]
    for idx, fur in enumerate(node.placement):
        fur['color'] = named_colors[idx % len(named_colors)]
        vis_furnitures_list.append(fur)
    try:
        img_np, textual_layout, coverage, idx_to_coor, emoji_use = create_grid(vis_furnitures_list, step=resolution, bound=node.bound, visualize=False, wall=node.wall, level='cell', draw_emoji=False, vis_size=1.0, draw_dir=True, render_size=render_size)
        if use_image:
            response = get_response_task(
                task='SCENE_EVALUATION', 
                **(prm_fmt(img_np, instruction) if prm else fmt(img_np, instruction)), 
                response_model=SceneEvaluationModel
            )
        else:
            response = get_response_task(
                task='SCENE_EVALUATION', 
                **(textual_prm_fmt(textual_layout, instruction) if prm else textual_fmt(textual_layout, instruction)), 
                response_model=SceneEvaluationModel
            )
        
        # Format the response properly
        result = {
            "physical_score": response['physical_score'] / 4,
            "semantic_score": response['semantic_score'] / 4,
            "layout_score": response['layout_score'] / 4,
            "instruction_alignment_score": response['instruction_alignment_score'] / 4,
            "physical_reasoning": response['physical_reasoning'],
            "semantic_reasoning": response['semantic_reasoning'],
            "layout_reasoning": response['layout_reasoning'],
            "instruction_alignment_reasoning": response['instruction_alignment_reasoning'],
            "average_score": (response['physical_score'] / 4 + response['semantic_score'] / 4 + response['layout_score'] / 4 + response['instruction_alignment_score'] / 4) / 4
        }
        return result
    except Exception as e:
        print(f"Error in scene evaluation: {e}")
        result = {
            "physical_score": 0,
            "semantic_score": 0, 
            "layout_score": 0,
            "instruction_alignment_score": 0,
            "physical_reasoning": "Evaluation failed",
            "semantic_reasoning": "Evaluation failed",
            "layout_reasoning": f"Evaluation failed: {str(e)}",
            "instruction_alignment_reasoning": "Evaluation failed",
            "average_score": 0
        }
        return result

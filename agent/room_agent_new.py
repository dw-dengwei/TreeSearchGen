from put.init_furniture import place_wall, place_center
from put.ori_policy import get_ori_policy
from put.offset_direction import get_offset_direction
from put.abs_orientation import get_abs_orientation
from put.offset import get_offset
from put.choose_side import get_offset_side
from utils.node import Node
from utils.load_json import load_json
from utils.visualize_layout_bottom import create_grid
from utils.logger import logger
from typing import Callable, Union
from random import choice, uniform
from collections import namedtuple
import math
import inspect, os
import numpy as np
from evaluator.evaluator import mcts_evaluator
import matplotlib.pyplot as plt
import networkx as nx
import PIL


def get_source_code_name_and_line(fn: Callable) -> str:
  try:
    return f"{os.path.basename(inspect.getsourcefile(fn))}:{inspect.getsourcelines(fn)[1]}"
  except:
    return 'unknown line'

class GlobalNode(Node, namedtuple('GLOBAL', 'placement bound wall anchor objects node_name win terminal attempt other_object_idx father_hash')): pass
class LocalNode(Node, namedtuple('LOCAL', """ori_policy offset_direction_candidate abs_orientation t_size axis alignment distance_desc fur_size relation_type related close_to_the_wall offset_direction_desc placement anchor fur bound wall resolution avoid_all draw_dir offset_1 offset_2 node_name win terminal attempt father_hash""")): pass

class DFS_solver:
  def __init__(self, name_to_fn: dict[str, Callable], name_to_max_attempts: dict[str, int], node_name_order: list[str]):
    self.name_to_fn = name_to_fn
    self.name_to_max_attempts = name_to_max_attempts 
    self.node_name_order = node_name_order
    self.solution_path = []
    logger.debug(f'{self.name_to_max_attempts=}')

  def _dfs(self, node: Union[GlobalNode, LocalNode], path: list = None):
    """Depth-first search process to find a terminal node in a tree structure.
    Args:
      node: The current Layout node to start or continue the search from.
      path: List tracking the current path being explored
    Returns:
      - tuple: (bool, list) - Success status and solution path if found
    """
    if path is None:
      path = []
    
    path.append(node)

    if node.terminal:
      if node.win:
        self.solution_path = path.copy()
        return True
      path.pop()
      return False
    
    children = node.find_children(self.name_to_fn, self.name_to_max_attempts, self.node_name_order)
    for idx, child in enumerate(children):
      status = self._dfs(child, path)
      if status:
        return True

    path.pop()
    return False

  def solve(self, root: Union[GlobalNode, LocalNode]) -> tuple[bool, list[Union[GlobalNode, LocalNode]]]:
    """Solve and return both success status and solution path.
    Args:
      root: Starting node for the search
    Returns:
      - tuple: (bool, list) - Success status and solution path if found
    """
    self.solution_path = []
    success = self._dfs(root)
    return success, self.solution_path

class MCTS_solver:
  def __init__(self, name_to_fn: dict[str, Callable], name_to_max_attempts: dict[str, int], node_name_order: list[str], exploration_weight: float = 1.0, max_iterations: int = 10, rollout_times: int = 10, vl_model=None, visualize_mcts=False, resolution=0.3, output_dir='./mcts_tree', prm_threshold=0.3, instruction=None, use_image=True):
    self.name_to_fn = name_to_fn
    self.name_to_max_attempts = name_to_max_attempts
    self.node_name_order = node_name_order
    self.exploration_weight = exploration_weight
    self.max_iterations = max_iterations
    self.rollout_times = rollout_times
    self.vl_model = vl_model  # External evaluator function
    self.solution_path = []
    self.visits = {}  # Store visit counts for nodes
    self.rewards = {}  # Store rewards for nodes
    self.prm_rewards = {}  # Store PRM rewards for nodes
    self.children = {}  # Store children for nodes
    self.visualize_mcts = visualize_mcts  # Flag to enable visualization
    self.mcts_graph = nx.DiGraph()  # Graph for visualization
    self.current_iteration = 0  # Track current iteration
    self.node_to_id = {}  # Map nodes to unique IDs for visualization
    self.next_node_id = 0  # Counter for generating unique node IDs
    self.resolution = resolution  # Resolution for grid visualization
    self.output_dir = output_dir  # Output directory for visualization
    self.prm_threshold = prm_threshold
    self.use_image = use_image
    os.makedirs(self.output_dir, exist_ok=True)
    if os.path.exists(self.output_dir):
      for file in os.listdir(self.output_dir):
        os.remove(os.path.join(self.output_dir, file))
    logger.debug(f'{self.name_to_max_attempts=}')
    self.instruction = instruction

  def _get_node_hash(self, node: Union[GlobalNode, LocalNode]) -> tuple:
    """Create a hashable representation of a node"""
    return node.__hash__()
    
  def _get_node_id(self, node: Union[GlobalNode, LocalNode]) -> int:
    """Get a unique ID for a node, creating one if it doesn't exist"""
    node_hash = self._get_node_hash(node)
    if node_hash not in self.node_to_id:
        self.node_to_id[node_hash] = self.next_node_id
        self.next_node_id += 1
    return self.node_to_id[node_hash]

  def _select(self, node: Union[GlobalNode, LocalNode]) -> Union[GlobalNode, LocalNode]:
    """Select a node to expand using UCB1 formula"""
    if self._get_node_hash(node) not in self.children or not self.children[self._get_node_hash(node)]:
      logger.debug(f"Select: No children for node {node.node_name} {hash(node)}, returning node")
      return node
    
    # UCB1 formula: exploitation + exploration
    def ucb1(n):
      node_hash = self._get_node_hash(n)
      if n.terminal and not n.win:
        return float('-inf')
      if self.prm_rewards[node_hash] < self.prm_threshold:
        return float('-inf')

      if node_hash not in self.visits or self.visits[node_hash] == 0:
        logger.debug(f"UCB1: Node {n.node_name} {hash(n)} not visited, returning inf")
        return float('inf')
      exploitation = self.rewards[node_hash] / self.visits[node_hash]
      exploration = self.exploration_weight * math.sqrt(math.log(self.visits[self._get_node_hash(node)]) / self.visits[node_hash])
      ucb_value = exploitation + exploration
      logger.debug(f"UCB1: Node {n.node_name} {hash(n)} - exploitation: {exploitation:.4f}, exploration: {exploration:.4f}, total: {ucb_value:.4f}")
      return ucb_value
    
    children = self.children[self._get_node_hash(node)]
    best_child = max(children, key=ucb1)
    
    # Visualization - highlight selection path
    if self.visualize_mcts:
      node_id = self._get_node_id(node)
      child_id = self._get_node_id(best_child)
      if self.mcts_graph.has_edge(node_id, child_id):
        self.mcts_graph[node_id][child_id]['color'] = 'red'
        self.mcts_graph[node_id][child_id]['width'] = 2.0
    
    logger.debug(f"Select: Chose child {best_child.node_name} {hash(best_child)} from {node.node_name} with {len(children)} children {list(map(lambda x: hash(x), children))}")
    return best_child

  def _expand(self, node: Union[GlobalNode, LocalNode]):
    """Expand a node by generating its children"""
    if node.terminal:
      logger.debug(f"Expand: Node {node.node_name} {hash(node)} is terminal, not expanding")
      return
    
    node_hash = self._get_node_hash(node)
    if node_hash not in self.children:
      children = list(node.find_children(self.name_to_fn, self.name_to_max_attempts, self.node_name_order))
      self.children[node_hash] = children
      for c in children:
        if self.prm_threshold >= 1:
          self.prm_rewards[self._get_node_hash(c)] = self.prm_threshold + 1
        else:
          self.prm_rewards[self._get_node_hash(c)] = mcts_evaluator(c, 0.3, 1.0, self.instruction, self.vl_model, prm=True, use_image=self.use_image)['average_score']
      
      # Visualization - add new children to graph
      if self.visualize_mcts:
        node_id = self._get_node_id(node)
        # Add node if not already in graph
        if node_id not in self.mcts_graph:
          self.mcts_graph.add_node(node_id, 
                                  label=f"{node.node_name}",
                                  hash=hash(node),
                                  father_hash=node.father_hash,
                                  placement=node.placement,
                                  bound=getattr(node, 'bound', [0, 10, 0, 10]),
                                  wall=getattr(node, 'wall', (0, 0, 0, 0)),
                                  prm_reward=self.prm_rewards.get(self._get_node_hash(node), 0),
                                  visits=0,
                                  reward=0.0,
                                  terminal=node.terminal,
                                  win=getattr(node, 'win', False))
        
        # Add children to graph
        for child in children:
          child_id = self._get_node_id(child)
          self.mcts_graph.add_node(child_id, 
                                  label=f"{child.node_name}",
                                  hash=hash(child),
                                  father_hash=child.father_hash,
                                  placement=child.placement,
                                  bound=getattr(child, 'bound', [0, 10, 0, 10]),
                                  wall=getattr(child, 'wall', (0, 0, 0, 0)),
                                  prm_reward=self.prm_rewards.get(self._get_node_hash(child), 0),
                                  visits=0,
                                  reward=0.0,
                                  terminal=child.terminal,
                                  win=getattr(child, 'win', False))
          self.mcts_graph.add_edge(node_id, child_id, color='black', width=1.0)
      
      logger.debug(f"Expand: Node {node.node_name} {hash(node)} expanded with {len(children)} children {list(map(lambda x: hash(x), children))}")


  def _simulate(self, node: Union[GlobalNode, LocalNode], depth: int = 0) -> float:
    """Simulate using an external evaluator or random actions to estimate node value
    
    Args:
      node: The node to evaluate
      depth: Current simulation depth
    
    Returns:
      float: A reward value between 0.0 and 1.0
    """
    # Terminal nodes have definite outcomes
    if node.terminal:
      if not node.win:
        # logger.debug(f"Simulate: Terminal node {node.node_name} with reward 0.0")
        return 0.0, {
          "physical_score": 0,
          "semantic_score": 0, 
          "layout_score": 0,
          "physical_reasoning": "Evaluation failed",
          "semantic_reasoning": "Evaluation failed",
          "layout_reasoning": f"Evaluation failed",
          "average_score": 0
        }

      reward_dict = mcts_evaluator(node, 0.3, 1.0, self.instruction, self.vl_model, use_image=self.use_image)
      reward = reward_dict['average_score']
      # logger.debug(f"Simulate: Terminal node {node.node_name} with reward {reward_dict}")
      return reward, reward_dict
    
    child_node = node.find_random_child(self.name_to_fn, self.name_to_max_attempts, self.node_name_order)
    
    # 为避免可视化过于复杂，我们不再在模拟阶段显示边
    # Visualization - highlight simulation path has been removed to simplify visualization
    
    # logger.debug(f"Simulate: Generated random child {child_node.node_name} at depth {depth}")
    return self._simulate(child_node, depth + 1)
  
  def _backpropagate(self, node: Union[GlobalNode, LocalNode], path: list[Union[GlobalNode, LocalNode]], reward: float):
    """Update statistics for nodes in the path"""
    logger.debug(f"Backpropagate: Updating path of length {len(path)} with reward {reward:.4f}")
    
    # Visualization - highlight backpropagation path
    if self.visualize_mcts:
      for i in range(len(path) - 1):
        parent_id = self._get_node_id(path[i])
        child_id = self._get_node_id(path[i+1])
        if self.mcts_graph.has_edge(parent_id, child_id):
          self.mcts_graph[parent_id][child_id]['color'] = 'green'
          self.mcts_graph[parent_id][child_id]['width'] = 1.5
    
    for n in path:
      node_hash = self._get_node_hash(n)
      if node_hash not in self.visits:
        self.visits[node_hash] = 0
        self.rewards[node_hash] = 0.0
      old_value = self.rewards[node_hash] / max(1, self.visits[node_hash])
      self.visits[node_hash] += 1
      self.rewards[node_hash] += reward
      new_value = self.rewards[node_hash] / self.visits[node_hash]
      
      # Update node attributes in visualization
      if self.visualize_mcts:
        node_id = self._get_node_id(n)
        if node_id in self.mcts_graph:
          self.mcts_graph.nodes[node_id]['visits'] = self.visits[node_hash]
          self.mcts_graph.nodes[node_id]['reward'] = new_value
      
      logger.debug(f"Backpropagate: Node {n.node_name} {hash(n)} - visits: {self.visits[node_hash]}, avg reward: {old_value:.4f} -> {new_value:.4f}")

  def _draw_mcts_tree(self, title=None):
    """Draw the MCTS tree visualization"""
    if not self.visualize_mcts:
      return
    
    # Limit the number of nodes to visualize to prevent oversized images
    if len(self.mcts_graph) > 50:
      logger.warning(f"Too many nodes in MCTS graph ({len(self.mcts_graph)}), pruning to 50 most important nodes")
      # Get most visited nodes
      node_visits = [(node_id, self.mcts_graph.nodes[node_id].get('visits', 0)) 
                    for node_id in self.mcts_graph.nodes]
      node_visits.sort(key=lambda x: x[1], reverse=True)
      important_nodes = [node_id for node_id, _ in node_visits[:50]]
      
      # Create a subgraph with only important nodes
      sub_graph = self.mcts_graph.subgraph(important_nodes).copy()
      # Remove disconnected nodes
      connected_nodes = list(nx.node_connected_component(sub_graph.to_undirected(), important_nodes[0]))
      sub_graph = sub_graph.subgraph(connected_nodes).copy()
      vis_graph = sub_graph
      logger.debug(f"Visualizing {len(vis_graph)} nodes after pruning")
    else:
      vis_graph = self.mcts_graph
    
    # Create two separate visualizations: the MCTS tree and the layout images
    
    # 1. First create the MCTS tree visualization
    plt.figure(figsize=(12, 8))
    
    # Define node positions using hierarchical layout
    try:
      pos = nx.nx_agraph.graphviz_layout(vis_graph, prog='dot')
    except Exception as e:
      logger.warning(f"Graphviz layout failed: {e}, falling back to spring layout")
      pos = nx.spring_layout(vis_graph, k=0.5, iterations=50)
    
    # Create a custom colormap for nodes based on visits and rewards
    node_colors = []
    node_sizes = []
    node_labels = {}
    
    max_visits = max([data['visits'] for _, data in vis_graph.nodes(data=True)]) if vis_graph.nodes else 1
    
    for node_id, data in vis_graph.nodes(data=True):
      # Calculate color based on reward (red for low, green for high)
      reward = data.get('reward', 0)
      # 确保reward在0-1范围内
      # reward = reward / 4
      
      visits = data.get('visits', 0)
      terminal = data.get('terminal', False)
      win = data.get('win', False)
      prm_reward = data.get('prm_reward', 0)
      
      # Size based on visit count (logarithmic scale to prevent huge nodes)
      size = 300 * (math.log(visits + 1) / math.log(max_visits + 1)) if max_visits > 0 else 100
      node_sizes.append(max(100, size))
      
      # Color based on reward and terminal status
      if terminal and win:
        node_colors.append('gold')  # Winning terminal nodes are gold
      elif terminal:
        node_colors.append('red')   # Losing terminal nodes are red
      else:
        # Non-terminal nodes colored by reward value - 使用固定的颜色值而不是元组
        if reward < 0.3:
            node_colors.append('red')
        elif reward < 0.6:
            node_colors.append('orange')
        else:
            node_colors.append('green')
      
      # Create labels with node info
      node_labels[node_id] = f"ID:{node_id}\n{data['label']}\nV:{visits}\nR:{(reward):.2f}\nPRM:{(prm_reward):.2f}\nHash:{data['hash']}\nFather Hash:{data['father_hash']}"
    
    # Get edge colors and widths
    edge_colors = [data['color'] for _, _, data in vis_graph.edges(data=True)]
    edge_widths = [data['width'] for _, _, data in vis_graph.edges(data=True)]
    edge_styles = [data.get('style', 'solid') for _, _, data in vis_graph.edges(data=True)]
    
    # Draw the graph
    nx.draw_networkx_nodes(vis_graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    # Draw edges with different styles
    solid_edges = [(u, v) for u, v, d in vis_graph.edges(data=True) if d.get('style', 'solid') == 'solid']
    dashed_edges = [(u, v) for u, v, d in vis_graph.edges(data=True) if d.get('style', 'solid') == 'dashed']
    
    solid_colors = [d['color'] for u, v, d in vis_graph.edges(data=True) if d.get('style', 'solid') == 'solid']
    dashed_colors = [d['color'] for u, v, d in vis_graph.edges(data=True) if d.get('style', 'solid') == 'dashed']
    
    solid_widths = [d['width'] for u, v, d in vis_graph.edges(data=True) if d.get('style', 'solid') == 'solid']
    dashed_widths = [d['width'] for u, v, d in vis_graph.edges(data=True) if d.get('style', 'solid') == 'dashed']
    
    nx.draw_networkx_edges(vis_graph, pos, edgelist=solid_edges, edge_color=solid_colors, width=solid_widths)
    nx.draw_networkx_edges(vis_graph, pos, edgelist=dashed_edges, edge_color=dashed_colors, 
                          width=dashed_widths, style='dashed')
    
    nx.draw_networkx_labels(vis_graph, pos, labels=node_labels, font_size=8)
    
    if title:
      plt.title(title)
    plt.axis('off')
    
    # Save the MCTS tree visualization
    tree_filename = os.path.join(self.output_dir, f"mcts_tree_iteration_{self.current_iteration}.png")
    plt.savefig(tree_filename, dpi=100, bbox_inches='tight')
    plt.close()
    
    # 2. Now create a grid of layout images for important nodes
    # Create layout visualizations for nodes
    node_layout_images = {}
    important_nodes = []
    
    # First, get the root node
    root_nodes = [n for n in vis_graph.nodes() if vis_graph.in_degree(n) == 0]
    if root_nodes:
        root_node = root_nodes[0]
        important_nodes.append(root_node)
    
    # Then get nodes along the most promising path (highest reward path)
    current = root_node if root_nodes else None
    while current and len(important_nodes) < 5:  # Get up to 5 nodes along the path
        successors = list(vis_graph.successors(current))
        if not successors:
            break
        # Choose successor with highest reward
        best_successor = max(successors, 
                            key=lambda n: vis_graph.nodes[n].get('reward', 0) / max(1, vis_graph.nodes[n].get('visits', 1)))
        important_nodes.append(best_successor)
        current = best_successor
    
    # Then add other important nodes based on visit count
    other_nodes = [(node_id, vis_graph.nodes[node_id].get('visits', 0)) 
                  for node_id in vis_graph.nodes 
                  if node_id not in important_nodes and 'placement' in vis_graph.nodes[node_id] 
                  and vis_graph.nodes[node_id]['placement']]
    other_nodes.sort(key=lambda x: x[1], reverse=True)
    
    # Add remaining important nodes to fill up to 9 total
    remaining_slots = 9 - len(important_nodes)
    for node_id, _ in other_nodes[:remaining_slots]:
        important_nodes.append(node_id)
    
    if not important_nodes:
      logger.warning("No important nodes with placement data found")
      return
    
    # Generate layout visualizations for important nodes
    for node_id in important_nodes:
      data = vis_graph.nodes[node_id]
      vis_furnitures_list = []
      named_colors = [
        'lightcoral', 'salmon', 'sienna', 'darkorange', 'gold', 'olive', 'yellow', 'yellowgreen', 'green', 'lightseagreen', 'deepskyblue', 'dodgerblue', 'blue', 'darkblue', 'indigo', 'purple', 'magenta', 'hotpink', 'pink', 'lightpink', 'white', 'lightgrey'
      ]
      for idx, fur in enumerate(data['placement']):
        fur_copy = fur.copy()
        fur_copy['color'] = named_colors[idx % len(named_colors)]
        vis_furnitures_list.append(fur_copy)
      
      # Get node information from the graph
      bound = data.get('bound', [0, 10, 0, 10])
      wall = data.get('wall', (0, 0, 0, 0))
      
      # Create grid visualization
      try:
        img_np, textual_layout, _, _, _ = create_grid(vis_furnitures_list, step=self.resolution, bound=bound, visualize=False, wall=wall, level='cell', draw_emoji=False, vis_size=1.0, draw_dir=True)
      except PIL.Image.DecompressionBombError:
        plt.close()
        return
      node_layout_images[node_id] = img_np
    
    # Create a figure for the layout grid
    rows = int(math.ceil(len(important_nodes) / 3))
    cols = min(3, len(important_nodes))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10 * rows / 3))
    
    # Make axes accessible for both single and multi-row cases
    if rows == 1 and cols == 1:
      axes = np.array([[axes]])
    elif rows == 1:
      axes = np.array([axes])
    elif cols == 1:
      axes = axes.reshape(-1, 1)
    
    # Add layout visualizations to the grid
    for i, node_id in enumerate(important_nodes):
      row = i // cols
      col = i % cols
      ax = axes[row, col]
      
      # Get node data
      data = vis_graph.nodes[node_id]
      img = node_layout_images[node_id]
      
      # Show image
      ax.imshow(img)
      
      # Add node info as title
      visits = data.get('visits', 0)
      reward = data.get('reward', 0)
      # / 4  # Same normalization as above
      node_name = data.get('label', f"Node {node_id}")
      ax.set_title(f"ID:{node_id}\n{node_name}\nVisits: {visits}, Reward: {reward:.2f}")
      
      # Remove axes
      ax.axis('off')
    
    # Hide any unused axes
    for i in range(len(important_nodes), rows * cols):
      row = i // cols
      col = i % cols
      axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save the layout grid visualization
    layouts_filename = os.path.join(self.output_dir, f"mcts_layouts_iteration_{self.current_iteration}.png")
    plt.savefig(layouts_filename, dpi=100, bbox_inches='tight')
    
    # For Jupyter Notebook display
    try:
      import IPython.display as display
      # Display both images in the notebook
      # display.display(display.Image(filename=tree_filename))
      # display.display(display.Image(filename=layouts_filename))
      logger.debug(f"Visualizations displayed and saved to {tree_filename} and {layouts_filename}")
    except ImportError:
      logger.debug(f"IPython display not available, visualizations saved to files")
    
    plt.close()

  def _mcts_search(self, root: Union[GlobalNode, LocalNode]) -> bool:
    """Main MCTS algorithm"""
    path = []
    logger.debug(f"MCTS Search: Starting with root node {root.node_name}, max iterations: {self.max_iterations}, rollout times: {self.rollout_times}")
    self.prm_rewards[self._get_node_hash(root)] = self.prm_threshold + 1.0
    
    # Initialize visualization
    if self.visualize_mcts:
      self.mcts_graph = nx.DiGraph()
      root_id = self._get_node_id(root)
      self.mcts_graph.add_node(root_id, 
                              label=f"{root.node_name}",
                              hash=hash(root),
                              father_hash=root.father_hash,
                              placement=root.placement,
                              bound=getattr(root, 'bound', [0, 10, 0, 10]),
                              wall=getattr(root, 'wall', (0, 0, 0, 0)),
                              prm_reward=1.0,
                              visits=0,
                              reward=0.0,
                              terminal=root.terminal,
                              win=getattr(root, 'win', False))
      # 初始可视化
      self._draw_mcts_tree(f"MCTS Tree - Initial State")
    
    # pbar = tqdm(range(self.max_iterations), desc="MCTS Search Iteration", leave=False)
    pbar = range(self.max_iterations)

    for iteration in pbar:
      self.current_iteration = iteration
      # Selection
      node = root
      path = [node]
      logger.debug(f"MCTS Search: Iteration {iteration+1}/{self.max_iterations}")
      
      selection_depth = 0
      # Reset edge colors for visualization
      if self.visualize_mcts:
        for u, v, data in self.mcts_graph.edges(data=True):
          data['color'] = 'black'
          data['width'] = 1.0
          data['style'] = 'solid'
      
      while self._get_node_hash(node) in self.children and self.children[self._get_node_hash(node)] and not node.terminal:
        node = self._select(node)
        path.append(node)
        selection_depth += 1
      
      # Expansion
      if not node.terminal:
        self._expand(node)
        if self.children[self._get_node_hash(node)]:
          child = choice(self.children[self._get_node_hash(node)])
          path.append(child)
          node = child
          logger.debug(f"MCTS Search: Added child {child.node_name} {hash(child)} to path")
          
          # Visualization - highlight expansion
          if self.visualize_mcts:
            parent_id = self._get_node_id(path[-2])
            child_id = self._get_node_id(child)
            if self.mcts_graph.has_edge(parent_id, child_id):
              self.mcts_graph[parent_id][child_id]['color'] = 'orange'
              self.mcts_graph[parent_id][child_id]['width'] = 1.5
            
            # 不再显示Expansion阶段的可视化
      
      # Simulation
      logger.debug(f"MCTS Search: Starting simulation from node {node.node_name} {hash(node)}")
      rewards = []
      for _ in range(self.rollout_times):
        reward, reward_dict = self._simulate(node)
        rewards.append(reward)
        logger.debug(f"MCTS Search: Rollout {_ + 1}/{self.rollout_times} complete, reward: {reward:.4f}, reward_physical: {reward_dict['physical_score']}, reward_semantic: {reward_dict['semantic_score']}, reward_layout: {reward_dict['layout_score']}, physical_reasoning: {reward_dict['physical_reasoning']}, semantic_reasoning: {reward_dict['semantic_reasoning']}, layout_reasoning: {reward_dict['layout_reasoning']}")
      reward = sum(rewards) / len(rewards)
      logger.debug(f"MCTS Search: Simulation complete, reward: {reward:.4f}")
      
      # 不再显示Simulation阶段的可视化
      
      # Backpropagation
      self._backpropagate(node, path, reward)
      
      # 在每次迭代结束后显示一次可视化，而不是每个阶段都显示
      if self.visualize_mcts:
        self._draw_mcts_tree(f"MCTS Tree - Iteration {iteration+1}")
      
      # Check if we found a winning node
      if node.terminal and node.win:
        logger.debug(f"MCTS Search: Found winning terminal node {node.node_name} {hash(node)} at iteration {iteration+1}")
        self.solution_path = path
        if self.visualize_mcts:
          self._draw_mcts_tree(f"MCTS Tree - Final (Found Winning Node)")
        return True
    
    # If we've exhausted iterations, select the best path
    logger.debug(f"MCTS Search: Max iterations reached, finding best path from {len(self.children.get(self._get_node_hash(root), []))} root children")
    if self._get_node_hash(root) in self.children and self.children[self._get_node_hash(root)]:
      # Log all child options with their stats
      for child in self.children[self._get_node_hash(root)]:
        child_hash = self._get_node_hash(child)
        visits = self.visits.get(child_hash, 0)
        avg_reward = self.rewards.get(child_hash, 0) / max(1, visits)
        logger.debug(f"Child option: {child.node_name} - visits: {visits}, avg reward: {avg_reward:.4f}")
      
      best_child = max(self.children[self._get_node_hash(root)], 
                      key=lambda n: self.rewards.get(self._get_node_hash(n), 0) / max(1, self.visits.get(self._get_node_hash(n), 0)))
      
      best_child_hash = self._get_node_hash(best_child)
      logger.debug(f"Selected best child: {best_child.node_name} with avg reward {self.rewards.get(best_child_hash, 0)/max(1, self.visits.get(best_child_hash, 0)):.4f}")
      
      # Recursively build the best path
      best_path = [root, best_child]
      current = best_child
      path_length = 1
      
      # Highlight best path in visualization
      if self.visualize_mcts:
        root_id = self._get_node_id(root)
        child_id = self._get_node_id(best_child)
        if self.mcts_graph.has_edge(root_id, child_id):
          self.mcts_graph[root_id][child_id]['color'] = 'purple'
          self.mcts_graph[root_id][child_id]['width'] = 3.0
      
      while self._get_node_hash(current) in self.children and self.children[self._get_node_hash(current)] and not current.terminal:
        children_options = self.children[self._get_node_hash(current)]
        # Log all child options with their stats
        for child in children_options:
          child_hash = self._get_node_hash(child)
          visits = self.visits.get(child_hash, 0)
          avg_reward = self.rewards.get(child_hash, 0) / max(1, visits)
          logger.debug(f"Path step {path_length+1} option: {child.node_name} - visits: {visits}, avg reward: {avg_reward:.4f}")
        
        next_node = max(children_options, 
                      key=lambda n: self.rewards.get(self._get_node_hash(n), 0) / max(1, self.visits.get(self._get_node_hash(n), 0)))
        
        best_path.append(next_node)
        
        # Highlight best path in visualization
        if self.visualize_mcts:
          parent_id = self._get_node_id(current)
          child_id = self._get_node_id(next_node)
          if self.mcts_graph.has_edge(parent_id, child_id):
            self.mcts_graph[parent_id][child_id]['color'] = 'purple'
            self.mcts_graph[parent_id][child_id]['width'] = 3.0
        
        current = next_node
        path_length += 1
        
        logger.debug(f"Added to best path: {current.node_name} (step {path_length})")
        
        if current.terminal and current.win:
          logger.debug(f"Found winning terminal node in best path construction: {current.node_name}")
          self.solution_path = best_path
          if self.visualize_mcts:
            self._draw_mcts_tree(f"MCTS Tree - Final (Best Path to Winning Node)")
          return True
      
      logger.debug(f"Best path construction complete, length: {len(best_path)}, terminal: {current.terminal}, win: {getattr(current, 'win', False)}")
      self.solution_path = best_path
      
      if self.visualize_mcts:
        self._draw_mcts_tree(f"MCTS Tree - Final (Best Available Path)")
    
    return False

  def solve(self, root: Union[GlobalNode, LocalNode]) -> tuple[bool, list[Union[GlobalNode, LocalNode]]]:
    """Solve and return both success status and solution path.
    Args:
      root: Starting node for the search
    Returns:
      - tuple: (bool, list) - Success status and solution path if found
    """
    logger.debug(f"MCTS Solve: Starting solve for root node {root.node_name}")
    self.solution_path = []
    self.visits = {}
    self.rewards = {}
    self.children = {}
    self.node_to_id = {}
    self.next_node_id = 0
    
    success = self._mcts_search(root)
    logger.debug(f"MCTS Solve: Search complete, success: {success}, path length: {len(self.solution_path)}")
    return success, self.solution_path

def get_consistent_response(fn, args, options, key, times=5, to_key=lambda x: x, to_value=lambda x: x):
	answers = [fn(*args)['output'] for _ in range(times)]
	vote = {}
	for ans in answers:
		tmp = to_key(ans[key])
		if tmp not in vote.keys():
			vote[tmp] = 1
		else:
			vote[tmp] += 1
	output = max(vote, key=vote.get)
	output = to_value(output)
	return output, answers

sort_key = [
	None,
	('center'),
	('left side', 'right side'),
	('left side', 'center', 'right side'),
	('left side', 'left center', 'right center', 'right side'),
	('left side', 'left center', 'center', 'right center', 'right side'),
]

def get_rank_fn(len):
	def get_rank(value):
		try:
			return sort_key[len].index(value)
		except Exception as e:
			raise e
	return get_rank

class RoomAgent:
  def __init__(self, scene_graph_path, tree_search_config, vl_model, language_model, use_solver, verbose, resolution=0.3, visualize_mcts=False, output_dir='./mcts_tree', instruction=None, prm_threshold=0.3, use_image=True):
    data = load_json(scene_graph_path)
    areas = [{
    'name': a['name'],
    'desc': a['desc'],
    'related_anchor_region': a['related_anchor_region'],
    'region_room_relation': a['region_room_relation'],
    'region_dimension': a['region_dimension'],
    'furnitures': a['furnitures']
    } for a in data['functional_area']]
    # anchor_region = None
    # for a in areas:
    #   if a['related_anchor_region'].lower() == a['name'].lower():
    #     anchor_region = a['name']
    # assert anchor_region is not None
    rank_fn = get_rank_fn(len(areas))
    areas = sorted(areas, key=lambda s: rank_fn(s['region_room_relation']))

    self.areas = areas
    self.resolution = resolution
    self.verbose = verbose
    self.room_desc = data['room_desc']
    self.room_dimension = data['room_dimension']
    self.get_vl_model_response = vl_model
    self.get_language_model_response = language_model
    self.tree_search_config = tree_search_config
    self.use_solver = use_solver
    self.visualize_mcts = visualize_mcts
    self.output_dir = output_dir
    self.instruction = instruction
    self.prm_threshold = prm_threshold
    self.use_image = use_image

  def is_exceed_room(self, rect):
    x1, y1 = rect['location']
    w1, h1, _ = rect['size']

    # 计算两个矩形的边界框
    rect_left = x1 - w1 / 2
    rect_right = x1 + w1 / 2
    rect_bottom = y1 - h1 / 2
    rect_top = y1 + h1 / 2

    if rect_left < 0 or rect_bottom < 0:
      return True
    if rect_right > self.room_dimension[0] or rect_top > self.room_dimension[1]:
      return True
    return False
  
  def is_overlap(self, rect1, rect2):
    """
    检查两个矩形是否有交叉。
    rect1 和 rect2 都是字典，包含 'location' 和 'size' 属性。
    'location' 表示底部中心的坐标 [center_x, center_y]。
    'size' 表示物体在 x 轴和 y 轴上的长度 [width, height]。
    """
    x1, y1 = rect1['location']
    w1, h1, _ = rect1['size']
    x2, y2 = rect2['location']
    w2, h2, _ = rect2['size']

    # 计算两个矩形的边界框
    rect1_left = x1 - w1 / 2
    rect1_right = x1 + w1 / 2
    rect1_bottom = y1 - h1 / 2
    rect1_top = y1 + h1 / 2

    rect2_left = x2 - w2 / 2
    rect2_right = x2 + w2 / 2
    rect2_bottom = y2 - h2 / 2
    rect2_top = y2 + h2 / 2

    # 判断是否有交叉
    if (rect1_left < rect2_right and rect1_right > rect2_left and
            rect1_bottom < rect2_top and rect1_top > rect2_bottom):
        return True
    return False

  def generate_area_param(self, x_start, a):
    bound = [x_start, x_start + a['region_dimension'][0], 0, self.room_dimension[1]]
    x_start += a['region_dimension'][0]
    if len(self.areas) == 1:
      wall = (1, 1, 1, 1)
    else:
      if a['region_room_relation'] == 'left side':
        wall = (1, 0, 1, 1)
      elif a['region_room_relation'] == 'right side':
        wall = (1, 1, 1, 0)
      else:
        wall = (1, 0, 1, 0)
      
    anchor = None
    other = []
    for fur in a['furnitures']:
      if fur['type'].lower() == 'anchor':
        anchor = fur
      else:
        other.append(fur)
    distance_to_order = {
      'far': 0,
      'near': 1,
      'adjacent_to': 2,
    }
    other.sort(
      key=lambda s: (
        1 if s['functionally_grouped'].lower() == 'yes' else 0,
        distance_to_order[s['distance']],
        s['size'][0] * s['size'][1]
      ),
      reverse=True
    )
    assert anchor is not None

    anchor.update({
      k: None for k, v in other[0].items() if k not in anchor.keys()
    })
    anchor.update({
      'location': None,
      'anchor': True,
      'orientation': None,
    })

    # Replace keys containing '-' with '_' in anchor
    anchor_keys = list(anchor.keys())
    for key in anchor_keys:
      if '-' in key:
        new_key = key.replace('-', '_')
        anchor[new_key] = anchor.pop(key)
    
    # Replace keys containing '-' with '_' in other
    for fur in other:
      fur_keys = list(fur.keys())
      for key in fur_keys:
        if '-' in key:
          new_key = key.replace('-', '_')
          fur[new_key] = fur.pop(key)

    return bound, x_start, wall, anchor, other

  def place_anchor(self, layout: GlobalNode, random_move=False) -> tuple[bool, GlobalNode]:
    anchor = layout.anchor
    wall = layout.wall
    bound = layout.bound

    anchor_front_side = anchor['frontal'].lower()
    max_size, min_size = max(anchor['size'][:2]), min(anchor['size'][:2])
    anchor['size'] = [max_size, min_size, anchor['size'][2]]
    if anchor_front_side == 'longer':
      anchor['size'][0], anchor['size'][1] = anchor['size'][0], anchor['size'][1]
    elif anchor_front_side == 'shorter':
      anchor['size'][0], anchor['size'][1] = anchor['size'][1], anchor['size'][0]
    else:
      exit(-1)
      # shuffle(anchor['size'][:2])

    if anchor['placement_rule'].lower() == 'place_wall' or anchor['placement_rule'].lower() == 'place_corner':
      threshold = choice([0.2, 0.5, 1.0])
      status, anchor_put_info = place_wall(
        wall,
        bound,
        anchor['size'],
        front_side=anchor_front_side,
        threshold=threshold,
      )
    elif anchor['placement_rule'].lower() == 'place_center':
      status, anchor_put_info = place_center(
        wall,
        bound,
        anchor['size'],
        front_side=anchor_front_side,
        threshold=None,
      )
    else:
      exit(-1)
    
    if not status:
      return False, GlobalNode(node_name=layout.node_name, win=layout.win, terminal=layout.terminal, attempt=layout.attempt, placement=[], bound=layout.bound, wall=layout.wall, anchor=anchor, objects=layout.objects, other_object_idx=layout.other_object_idx + 1, father_hash=layout.father_hash)

    result = {
      'location': [anchor_put_info['output']['center_x'], anchor_put_info['output']['center_y']],
      'size': anchor_put_info['output']['size'],
      'name': anchor['name'],
      'anchor': True,
      'description': anchor['description'],
      'orientation': anchor_put_info['output']['orientation'],
    }
    anchor.update(result)
    placement = layout.placement + [result]
    status = True

    return status, GlobalNode(node_name=layout.node_name, win=layout.win, terminal=layout.terminal, attempt=layout.attempt, placement=placement, bound=layout.bound, wall=layout.wall, anchor=anchor, objects=layout.objects, other_object_idx=layout.other_object_idx + 1, father_hash=layout.father_hash)

  def place_others(self, layout: GlobalNode, random_move=False) -> tuple[bool, GlobalNode]:
    fur_list = layout.objects
    fur_idx = int(layout.other_object_idx)
    assert fur_idx >=0 and fur_idx < len(fur_list)
    anchor = layout.anchor
    fur = fur_list[fur_idx]
    fur_size = fur['size']
    fur_size = self.adjust_size(fur)
    fur['size'] = fur_size
    if random_move:
      # First randomly generate orientation: 0:top, 1:right, 2:bottom, 3:left
      orientation = choice([0, 1, 2, 3])
      
      # Adjust furniture size based on orientation
      # If orientation is right (1) or left (3), swap width and height
      if orientation in [1, 3]:
          adjusted_size = [fur_size[1], fur_size[0], fur_size[2]]
      else:
          adjusted_size = fur_size.copy()
      
      # Calculate valid location within scene boundaries
      x_min = layout.bound[0] + adjusted_size[0]/2
      x_max = layout.bound[1] - adjusted_size[0]/2
      y_min = layout.bound[2] + adjusted_size[1]/2
      y_max = layout.bound[3] - adjusted_size[1]/2
      
      # Ensure boundaries are valid (in case furniture is too large)
      if x_min >= x_max:
          x_min = x_max = (layout.bound[0] + layout.bound[1]) / 2
      if y_min >= y_max:
          y_min = y_max = (layout.bound[2] + layout.bound[3]) / 2
          
      return True, GlobalNode(
        terminal=layout.terminal,
        win=layout.win,
        node_name=layout.node_name,
        placement=layout.placement + [{
          'location': [uniform(x_min, x_max), uniform(y_min, y_max)],
          'size': adjusted_size,
          'name': fur['name'],
          'anchor': False,
          'description': fur['description'],
          'orientation': orientation
        }],
        bound=layout.bound,
        wall=layout.wall,
        anchor=layout.anchor,
        objects=layout.objects,
        attempt=layout.attempt,
        other_object_idx=layout.other_object_idx + 1,
        father_hash=layout.father_hash
      )

    other_place_attributes = self.generate_place_attributes(anchor, fur, self.resolution, fur_size)
    related = other_place_attributes['related']
    alignment = other_place_attributes['alignment']
    axis = other_place_attributes['axis']
    placement_rule = fur['placement_rule']

    if alignment == 'side alignment' and related == 'yes' and placement_rule != 'place_around':
      other_place_attributes['close_to_the_wall'] = 'no'
      solver = self._create_local_solver()
      root = self._create_root_node(fur, layout.anchor, layout, other_place_attributes)
      status, solution_path = solver.solve(root)
      return self._validate_and_create_result(solution_path, layout, axis, layout.anchor) if status else (False, layout)

    elif alignment == 'center alignment' and related == 'yes' and placement_rule != 'place_around':
      solver = self._create_local_solver()
      root = self._create_root_node(fur, anchor, layout, other_place_attributes)
      status, solution_path = solver.solve(root)
      return self._validate_and_create_result(solution_path, layout, axis, anchor) if status else (False, layout)

    else:  # not related or around alignment
      solver = self._create_local_solver(include_around_alignment=True)
      root = self._create_root_node(fur, anchor, layout, other_place_attributes, avoid_all=False)
      status, solution_path = solver.solve(root)
      return self._validate_and_create_result(solution_path, layout, axis, anchor) if status else (False, layout)

  def _create_local_solver(self, include_around_alignment=False):
    name_to_fn = {
      'generate_direction_desc': self.generate_direction_desc,
      'generate_draw_dir': self.generate_draw_dir,
      'generate_offset': self.generate_offset,
    }
    name_to_max_attempts = {
      'generate_direction_desc': self.tree_search_config.tree_width.floor.others.direction,
      'generate_draw_dir': self.tree_search_config.tree_width.floor.others.draw_dir,
      'generate_offset': self.tree_search_config.tree_width.floor.others.offset,
    }
    node_name_order = ['__start__', 'generate_direction_desc', 'generate_draw_dir', 'generate_offset']
    
    if include_around_alignment:
      name_to_fn['generate_center_x_or_y_around_alignment'] = self.generate_center_x_or_y_around_alignment
      name_to_max_attempts['generate_center_x_or_y_around_alignment'] = self.tree_search_config.tree_width.floor.others.center_x_or_y_around_alignment
      node_name_order.append('generate_center_x_or_y_around_alignment')
    
    return DFS_solver(
      name_to_fn=name_to_fn,
      name_to_max_attempts=name_to_max_attempts,
      node_name_order=node_name_order,
    )

  def _create_root_node(self, fur, anchor, layout, other_place_attributes, avoid_all=True):
    return LocalNode(
      fur=fur,
      anchor=anchor,
      placement=layout.placement,
      bound=layout.bound,
      wall=layout.wall,
      resolution=self.resolution,
      avoid_all=avoid_all,
      attempt=layout.attempt,
      **other_place_attributes,
      node_name='__start__',
      win=False,
      terminal=False,
      offset_direction_desc=None,
      draw_dir=None,
      offset_1=None,
      offset_2=None,
      father_hash=0,
    )

  def _validate_and_create_result(self, solution_path, layout, axis, anchor):
    offset_1 = solution_path[-1].offset_1
    offset_2 = solution_path[-1].offset_2 if hasattr(solution_path[-1], 'offset_2') else None
    fur_size = solution_path[-1].fur_size

    if offset_2 is None:  # side or center alignment
      if axis == 'row':
        center_x = anchor['location'][0]
        center_y = sum(offset_1) / len(offset_1)
      else:
        center_y = anchor['location'][1]
        center_x = sum(offset_1) / len(offset_1)
    else:  # around alignment
      if axis == 'row':
        center_x = sum(offset_2) / len(offset_2)
        center_y = sum(offset_1) / len(offset_1)
      else:
        center_x = sum(offset_1) / len(offset_1)
        center_y = sum(offset_2) / len(offset_2)

    result = {
      'location': [center_x, center_y],
      'size': fur_size,
      'name': solution_path[-1].fur['name'],
      'anchor': False,
      'description': solution_path[-1].fur['description'],
      'orientation': solution_path[-1].abs_orientation
    }

    # Validate placement
    for other_fur in layout.placement:
      if self.is_overlap(result, other_fur):
        return False, layout
        # raise Exception(f"Overlap detected: {result} and {other_fur}")
      if self.is_exceed_room(result):
        return False, layout
        # raise Exception(f"Exceed room: {result}")

    placement = layout.placement + [result]

    return True, GlobalNode(
      terminal=layout.terminal,
      win=layout.win,
      node_name=layout.node_name,
      placement=placement,
      bound=layout.bound,
      wall=layout.wall,
      anchor=layout.anchor,
      objects=layout.objects,
      attempt=layout.attempt,
      other_object_idx=layout.other_object_idx + 1,
      father_hash=layout.father_hash
    )

  @staticmethod
  def visualize_result(results, resolution):
    vis_furnitures_list = []
    named_colors = [
      'lightcoral', 'salmon', 'sienna', 'darkorange', 'gold', 'olive', 'yellow', 'yellowgreen', 'green', 'lightseagreen', 'deepskyblue', 'dodgerblue', 'blue', 'darkblue', 'indigo', 'purple', 'magenta', 'hotpink', 'pink', 'lightpink', 'white', 'lightgrey'
    ]
    idx = 0
    for area in results['areas']:
      # 检查object_list是否为空
      if not area['object_list']:
        continue
        
      for fur in area['object_list']:
        fur['color'] = named_colors[idx % len(named_colors)]
        vis_furnitures_list.append(fur)
        idx += 1
    room_width, room_height, _ = results['room_dimension']
    bound = [0, room_width, 0, room_height]
    wall = (0,0,0,0)
    img_np, textual_layout, coverage, idx_to_coor, emoji_use = create_grid(vis_furnitures_list, step=resolution, bound=bound, visualize=True, wall=wall, level='cell', draw_emoji=False, vis_size=1.0, draw_dir=True)

    return img_np
  
  
  def generate_offset_direction_candidate(self, anchor_orientation, relation_type):
    return get_offset_direction(
      anchor_orientation,
      relation_type
    )

  def generate_self_abs_orientation_candidate(self, offset_direction, ori_policy, anchor_orientation):
    return get_abs_orientation(
      offset_direction, ori_policy, anchor_orientation
    )

  def generate_self_abs_orientation(self, offset_direction, ori_policy, anchor_orientation):
    return self.generate_self_abs_orientation_candidate(offset_direction, ori_policy, anchor_orientation)

  def generate_self_size(self, self_abs_orientation, fur_size):
    if self_abs_orientation[0] in [1, 3]:
      fur_size = [fur_size[1], fur_size[0], fur_size[2]]
    return fur_size

  def generate_relation_type(self, placement_rule):
    relation_type = None
    if placement_rule.lower() == 'place_beside':
      relation_type = [1, 3]
    elif placement_rule.lower() == 'place_front':
      relation_type = 0
    elif placement_rule.lower() == 'place_around':
      relation_type = choice([[0, 2], [1, 3]])
    return relation_type 

  def generate_ori_policy(self, anchor_name, anchor_desc, fur_name, fur_desc):
    ori_policy, _ = get_consistent_response(
      fn=get_ori_policy,
      args=(anchor_name, anchor_desc, fur_name, fur_desc, self.get_language_model_response),
      options=['A', 'B', 'C', 'D'],
      key='orientation_rule',
      times=1
    )
    return ori_policy

  def generate_gridsize_offsetaxis(self, fur_size, resolution, offset_direction_candidate):
    t_size = math.ceil(fur_size[1] / resolution), math.ceil(fur_size[0] / resolution)
    if offset_direction_candidate[0] in [0, 2]:
      axis = 'row'
      t_size = t_size[0]
    elif offset_direction_candidate[0] in [1, 3]:
      axis = 'column'
      t_size = t_size[1]

    return t_size, axis

  def summarize_grid(self, existing_furnitures, axis, anchor_name):
    existing_coverage = []
    anchor_coverage = None
    for idx, obj in enumerate(existing_furnitures):
      ori = obj['orientation']
      ori_to_str = ['top', 'right', 'bottom', 'left']
      ori = ori_to_str[ori]
      s = f"""- {obj['name']}, covering the following {axis}s: {", ".join(obj['coverage'][axis][idx])}, face to {ori}."""
      if obj['name'] == anchor_name:
        anchor_coverage = obj['coverage'][axis][idx]
      existing_coverage.append(s)
    existing_coverage = "\n".join(existing_coverage)
    return existing_coverage, anchor_coverage

  def generate_center_xy_side_alignment(self, anchor_orientation, anchor_center_x, anchor_center_y, anchor_size, fur_size, axis):
    if anchor_orientation == 0:
      side_coor = anchor_center_y - anchor_size[1] / 2
      center_other = side_coor + fur_size[1] / 2
    elif anchor_orientation == 1:
      side_coor = anchor_center_x - anchor_size[0] / 2
      center_other = side_coor + fur_size[0] / 2
    elif anchor_orientation == 2:
      side_coor = anchor_center_y + anchor_size[1] / 2
      center_other = side_coor - fur_size[1] / 2
    elif anchor_orientation == 3:
      side_coor = anchor_center_x + anchor_size[0] / 2
      center_other = side_coor - fur_size[0] / 2
    return center_other

  def generate_center_x_or_y_around_alignment(self, node: LocalNode, random_move=False) -> tuple[bool, LocalNode]:

    anchor_name = node.anchor['name']
    fur_name = node.fur['name']
    t_size = math.ceil(node.fur_size[1] / node.resolution), math.ceil(node.fur_size[0] / node.resolution)
    if 'top' in node.offset_direction_desc or 'bottom' in node.offset_direction_desc:
      t_size = t_size[1]
    elif 'left' in node.offset_direction_desc or 'right' in node.offset_direction_desc:
      t_size = t_size[0]
    if 'adjacent_to' == node.distance_desc:
      adj_anchor = t_size
    else:
      adj_anchor = -1
    img_np, textual_layout, coverage, idx_to_coor, emoji_used = create_grid(node.placement, step=node.resolution, bound=node.bound, visualize=self.verbose, wall=node.wall, level='row' if node.axis=='column' else 'column', draw_emoji=True, direction_show=node.draw_dir, avoid_anchor=False, required_coor={'axis': node.axis, 'coor': node.offset_1}, adj_anchor=adj_anchor)

    if random_move:
      local_node = node._asdict()
      axis_2_offset = [v for k, v in emoji_used.items()]
      axis_2_offset.sort()
      start_idx = choice(range(len(axis_2_offset) - t_size + 1))
      axis_2_offset = axis_2_offset[start_idx:start_idx + t_size]
      local_node.update({
        'offset_2': axis_2_offset
      })
      local_node = LocalNode(**local_node)
      return True, local_node

    if self.use_image:
      axis_2_offset_name = get_offset(
        node.offset_direction_desc, 'row' if node.axis=='column' else 'column', anchor_name, node.close_to_the_wall, fur_name, t_size, emoji_used, node.distance_desc, img_np, textual_layout, self.get_vl_model_response, self.use_image
      )['output']['answer']
      try:
          axis_2_offset = [emoji_used[name] for name in axis_2_offset_name]
      except KeyError:
        return False, node
    else:
      axis_2_offset = get_offset(
        node.offset_direction_desc, 'row' if node.axis=='column' else 'column', anchor_name, node.close_to_the_wall, fur_name, t_size, emoji_used, node.distance_desc, img_np, textual_layout, self.get_vl_model_response, self.use_image
      )['output']['answer']
      if axis_2_offset == "None":
        return False, node
      axis_2_offset = [axis_2_offset]

    local_node = node._asdict()
    local_node.update({
      'offset_2': axis_2_offset
    })
    local_node = LocalNode(**local_node)
    return True, local_node

  def generate_direction_desc(self, node: LocalNode, random_move=False) -> tuple[bool, LocalNode]:
    anchor = node.anchor
    fur = node.fur
    placement = node.placement
    resolution = node.resolution
    bound = node.bound
    wall = node.wall
    axis = node.axis
    offset_direction_candidate = node.offset_direction_candidate
    avoid_all = node.avoid_all
    t_size = node.t_size
    ori_policy = node.ori_policy

    anchor_name = anchor['name']
    anchor_orientation = anchor['orientation']
    fur_name = fur['name']

    dir_to_desc = ['at the top of', 'right of', 'at the bottom of', 'left of']
    direction_desc_candidate = [dir_to_desc[dir] for dir in offset_direction_candidate]
    if random_move:
      direction_desc = choice(direction_desc_candidate)
      abs_orientation = self.generate_self_abs_orientation(dir_to_desc.index(direction_desc), ori_policy, anchor_orientation)[0]
      local_node = node._asdict()
      local_node.update({
        'offset_direction_desc': direction_desc,
        'abs_orientation': abs_orientation
      })
      local_node = LocalNode(**local_node)
      return True, local_node
    
    if len(direction_desc_candidate) == 1:
      direction_desc = direction_desc_candidate[0]
    else:
      try:
        img_np, textual_layout, coverage, idx_to_coor, emoji_used = create_grid(placement, step=resolution, bound=bound, visualize=self.verbose, wall=wall, level=axis, draw_emoji=True, avoid_all=avoid_all)
      except PIL.Image.DecompressionBombError:
        return False, node

      if len(emoji_used) != 0:
        tmp = [(k, v) for k, v in emoji_used.items()]
        tmp.sort(key=lambda x: x[1])
        segments = []
        prev = 0
        for idx in range(1, len(tmp)):
          if abs(abs(tmp[idx][1] - tmp[idx-1][1]) - resolution) > 1e-6:
            segment_length = idx - prev
            if segment_length >= t_size:
              segments.append(tmp[prev:idx])
            prev = idx
        if len(tmp) - prev >= t_size:
          segments.append(tmp[prev:])

        emoji_used = {}
        for seg in segments:
          for (k, v) in seg:
            emoji_used[k] = v

      direction_desc, _ = get_consistent_response(
        fn=get_offset_side,
        args=(direction_desc_candidate, axis, anchor_name, fur_name, t_size, img_np, textual_layout, self.get_vl_model_response, self.use_image),
        options=direction_desc_candidate,
          # + ["None"],
        key='side',
        times=1
      )
      logger.debug(f"Candidate: {direction_desc_candidate}, result: {direction_desc}")
      if direction_desc not in direction_desc_candidate:
        return False, node
        # raise Exception(f"Failed to generate direction_desc: {direction_desc} is not in {direction_desc_candidate}")
    abs_orientation = self.generate_self_abs_orientation(dir_to_desc.index(direction_desc), ori_policy, anchor_orientation)[0]

    local_node = node._asdict()
    local_node.update({
      'offset_direction_desc': direction_desc,
      'abs_orientation': abs_orientation
    })
    local_node = LocalNode(**local_node)
    return True, local_node

  def generate_draw_dir(self, node: LocalNode, random_move=False) -> tuple[bool, LocalNode]:
    direction_desc = node.offset_direction_desc
    draw_dir = -1
    if 'top' in direction_desc:
      draw_dir = 0
    elif 'right' in direction_desc:
      draw_dir = 1
    elif 'bottom' in direction_desc:
      draw_dir = 2
    elif 'left' in direction_desc:
      draw_dir = 3
    local_node = node._asdict()
    local_node.update({
      'draw_dir': draw_dir
    })
    local_node = LocalNode(**local_node)
    return True, local_node
  
  def generate_offset(self, node: LocalNode, random_move=False) -> tuple[bool, LocalNode]:
    draw_dir = node.draw_dir
    direction_desc = node.offset_direction_desc
    placement = node.placement
    resolution = node.resolution
    bound = node.bound
    wall = node.wall
    axis = node.axis
    anchor = node.anchor
    fur = node.fur
    t_size = node.t_size
    distance_desc = node.distance_desc
    close_to_the_wall = node.close_to_the_wall
    avoid_all = node.avoid_all
    anchor_name = anchor['name']
    fur_name = fur['name']

    if 'adjacent_to' == distance_desc:
      adj_anchor = t_size
    else:
      adj_anchor = -1
    img_np, textual_layout, coverage, idx_to_coor, emoji_used = create_grid(placement, step=resolution, bound=bound, visualize=self.verbose, wall=wall, level=axis, draw_emoji=True, direction_show=draw_dir, avoid_all=avoid_all, adj_anchor=adj_anchor)
    if len(emoji_used) != 0:
      tmp = [(k, v) for k, v in emoji_used.items()]
      tmp.sort(key=lambda x: x[1])
      segments = []
      prev = 0
      for idx in range(1, len(tmp)):
        if abs(abs(tmp[idx][1] - tmp[idx-1][1]) - resolution) > 1e-6:
          segment_length = idx - prev
          if segment_length >= t_size:
            segments.append(tmp[prev:idx])
          prev = idx
      if len(tmp) - prev >= t_size:
        segments.append(tmp[prev:])

      emoji_used = {}
      for seg in segments:
        for (k, v) in seg:
          emoji_used[k] = v
    

    if random_move:
      local_node = node._asdict()
      offset = [v for k, v in emoji_used.items()]
      offset.sort()
      start_idx = choice(range(len(offset) - t_size + 1))
      offset = offset[start_idx:start_idx + t_size]
      local_node.update({
        'offset_1': offset
      })
      local_node = LocalNode(**local_node)
      return True, local_node

    if len(emoji_used) < t_size:
      if self.verbose:
        pass
        # pprint(axis)
        # pprint(node._asdict())
      return False, node
      # raise Exception(f"Failed to generate offset: len(emoji_used)={len(emoji_used)} < t_size={t_size}")
    if close_to_the_wall == 'yes':
      if 'top' in direction_desc or 'right' in direction_desc:
        offset = [v for k, v in emoji_used.items()]
        offset.sort(reverse=True)
      elif 'bottom' in direction_desc or 'left' in direction_desc:
        offset = [v for k, v in emoji_used.items()]
        offset.sort()
      offset = offset[:t_size]
    else:
      if self.use_image:
        offset_name = get_offset(
          direction_desc, axis, anchor_name, close_to_the_wall, fur_name, t_size, emoji_used, distance_desc, img_np, textual_layout, self.get_vl_model_response, self.use_image
        )['output']['answer']
        try:
          offset = [emoji_used[name] for name in offset_name]
        except KeyError:
          return False, node
      else:
        offset = get_offset(
          direction_desc, axis, anchor_name, close_to_the_wall, fur_name, t_size, emoji_used, distance_desc, img_np, textual_layout, self.get_vl_model_response, self.use_image
        )['output']['answer']
        if offset== "None":
          return False, node
        offset = [offset]

    local_node = node._asdict()
    local_node.update({
      'offset_1': offset
    })
    local_node = LocalNode(**local_node)
    return True, local_node

  def adjust_size(self, fur):
    init_size = fur['size']

    front_side_output = fur['frontal'].lower()
    max_size, min_size = max(init_size[:2]), min(init_size[:2])
    if front_side_output == 'longer':
      fur_size = [max_size, min_size]
    elif front_side_output == 'shorter':
      fur_size = [max_size, min_size]
    else:
      exit(-1)
    fur_size.append(init_size[2])

    return fur_size

  def generate_place_attributes(self, anchor, fur, resolution, fur_size):
    alignment = fur['alignment']
    related = fur['functionally_grouped']
    distance_desc = fur['distance']
    relation_type = self.generate_relation_type(fur['placement_rule'])
    if related == 'yes' and relation_type != 0:
      close_to_the_wall = 'no'
    else:
      close_to_the_wall = fur['close_to_the_wall'].lower()

    if fur['functionally_grouped'].lower() == 'yes':
      if fur['placement_rule'].lower() == 'place_around':
        ori_policy = 'A'
      else:
        ori_policy = self.generate_ori_policy(anchor['name'], anchor['description'], fur['name'], fur['description'])
    else:
      if close_to_the_wall == 'yes':
        if 'adjacent_to' == distance_desc:
          alignment, ori_policy, related = ('side alignment', 'D', 'yes')
        elif 'near' == distance_desc:
          alignment, ori_policy, related = choice([
            ('side alignment', 'D', 'yes'),
            ('side alignment', 'A', 'no'),
          ])
        else:
          alignment, ori_policy, related = ('side alignment', 'A', 'no')
      else:
        ori_policy = 'A'
    offset_direction_candidate = self.generate_offset_direction_candidate(anchor['orientation'], relation_type)
    abs_orientation = self.generate_self_abs_orientation_candidate(offset_direction_candidate, ori_policy, anchor['orientation'])
    fur_size = self.generate_self_size(abs_orientation, fur_size)
    t_size, axis = self.generate_gridsize_offsetaxis(fur_size, resolution, offset_direction_candidate)
    # alignment = self.generate_alignment_policy(anchor['name'], anchor['description'], fur['name'], fur['description'])
    result = {
      'ori_policy': ori_policy,
      'offset_direction_candidate': offset_direction_candidate,
      'abs_orientation': abs_orientation,
      't_size': t_size,
      'axis': axis,
      'alignment': alignment,
      'distance_desc': distance_desc,
      'fur_size': fur_size,
      'relation_type': relation_type,
      'related': related,
      'close_to_the_wall': close_to_the_wall
    }

    return result

  def _gen_area(self, a, x_start):
    bound, x_start, wall, anchor, other = self.generate_area_param(x_start, a)
    name_to_fn = {'place_anchor': self.place_anchor}
    name_to_max_attempts = {'place_anchor': self.tree_search_config.tree_width.floor.anchor}
    node_name_order = ['__start__', 'place_anchor']

    name_to_fn.update({
      f"place_others_{fur['name']}_{fur_idx}": self.place_others
      for fur_idx, fur in enumerate(other)
    })
    name_to_max_attempts.update({
      f"place_others_{fur['name']}_{fur_idx}": self.tree_search_config.tree_width.floor.others.self
      for fur_idx, fur in enumerate(other)
    })
    node_name_order.extend([
      f"place_others_{fur['name']}_{fur_idx}" for fur_idx, fur in enumerate(other)
    ])

    logger.debug(f'name_to_fn: {name_to_fn}')
    logger.debug(f'name_to_max_attempts: {name_to_max_attempts}')
    logger.debug(f'node_name_order: {node_name_order}')

    if self.use_solver == 'dfs':
      solver = DFS_solver(
        name_to_fn=name_to_fn,
        name_to_max_attempts=name_to_max_attempts,
        node_name_order=node_name_order,
      )
    elif self.use_solver == 'mcts':
      solver = MCTS_solver(
        name_to_fn=name_to_fn,
        name_to_max_attempts=name_to_max_attempts,
        node_name_order=node_name_order,
        vl_model=self.get_vl_model_response,
        visualize_mcts=self.visualize_mcts,
        resolution=self.resolution,
        max_iterations=10,
        rollout_times=5,
        output_dir=os.path.join(self.output_dir, 'furniture_mcts', a['name']),
        exploration_weight=0.2,
        instruction=self.instruction,
        prm_threshold=self.prm_threshold,
        use_image=self.use_image
      )
    else:
      raise NotImplementedError(f'Invalid solver: {self.use_solver}')

    first_node_name = '__start__'
    
    root = GlobalNode(node_name=first_node_name, terminal=False, win=False, placement=[], bound=bound, wall=wall, anchor=anchor, objects=other, attempt=0, other_object_idx=-1, father_hash=0)
    status, solution_path = solver.solve(root)

    return status, solution_path, x_start

  def forward(self):
    x_start = 0
    logger.debug(" | ".join([a['name'] for a in self.areas]))
    results = {
      'areas': []
    }
    for a in self.areas[:]:
      logger.debug(f'Processing area [{a["name"]}]')
      status, solution_path, x_start = self._gen_area(a, x_start)
      if status and solution_path:
        results['areas'].append({
          "x_start": x_start,
          "object_list": solution_path[-1].placement
        })
      else:
        logger.debug(f'Failed to generate area {a["name"]}')
        # 添加一个空的对象列表，避免索引错误
        if len(solution_path) == 0:
          results['areas'].append({
            "x_start": x_start,
            "object_list": []
          })
        else:
          results['areas'].append({
            "x_start": x_start,
            "object_list": solution_path[-1].placement
          })


    results['room_dimension'] = self.room_dimension

    return results

def main(use_solver='mcts', visualize_mcts=False):
  import os
  from easydict import EasyDict as edict
  import yaml
  from utils.get_qwen import get_response_fn, get_api
  import matplotlib.pyplot as plt


  output_dir = '/home/dw/code/llm-scene/output/style_txt/4_2025-04-29_10:58_small_bedroom'
  furniture_resolution = 0.3
  tree_search_config = edict(yaml.safe_load(open('/home/dw/code/llm-scene/config/config.yaml', 'r')))
  vl_model_name = os.environ['VL_MODEL_NAME']
  vl_model_key = os.environ['VL_MODEL_KEY']
  vl_model_url = os.environ['VL_MODEL_URL']

  language_model_name = os.environ['LANGUAGE_MODEL_NAME']
  language_model_key = os.environ['LANGUAGE_MODEL_KEY']
  language_model_url = os.environ['LANGUAGE_MODEL_URL']
  log_name = os.path.join(output_dir, 'log.ansi')

  language_model = get_response_fn(
    get_api(api_key=language_model_key, api_url=language_model_url),
    is_vl_model=False,
    model=language_model_name,
    max_retry=5,
    verbose=log_name,
    system_prompt='You are a helpful AI assistant.'
  )

  vl_model = get_response_fn(
    get_api(api_key=vl_model_key, api_url=vl_model_url),
    is_vl_model=True,
    model=vl_model_name,
    max_retry=2,
    verbose=log_name,
    system_prompt='You are a helpful AI assistant.'
  )

  room_agent = RoomAgent(
    os.path.join(output_dir, '10_full_scene_graph.json'),
    tree_search_config=tree_search_config,
    vl_model=vl_model('precise'),
    language_model=language_model('precise'),
    verbose=False,
    resolution=furniture_resolution,
    use_solver=use_solver,
    visualize_mcts=visualize_mcts
  )
  results = room_agent.forward()
  try:
    img = RoomAgent.visualize_result(results, room_agent.resolution)
    plt.imsave(os.path.join(output_dir, '11_furniture_layout.png'), img)
  except PIL.Image.DecompressionBombError as e:
    print(f"Error: {e}")
  # print(results)
  return results

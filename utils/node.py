from abc import ABC, abstractmethod
from utils.logger import logger
from typing import Dict, List, Any, Callable, Generator
import json
import inspect
import os
import copy

def get_source_code_name_and_line(fn: Callable) -> str:
  try:
    return f"{os.path.basename(inspect.getsourcefile(fn))}:{inspect.getsourcelines(fn)[1]}"
  except:
    return 'unknown line'

class Node(ABC):
  """
  A representation of a single state.
  MCTS and ToT works by constructing a tree of these Nodes.
  """
  __abstractmethods__ = {'find_random_child', 'reward'}

  def __new__(cls, *args, **kwargs):
    if cls is Node:
      raise TypeError("Cannot instantiate abstract class Node directly")
    return super().__new__(cls, *args, **kwargs)

  def find_children(self, name_to_fn: Dict[str, Callable], name_to_max_attempts: Dict[str, int], node_name_order: List[str]) -> Generator['Node', None, None]:
    if self.terminal: # If the placement is finished then no moves can be made
      return set()
    child_node_name = node_name_order[node_name_order.index(self.node_name) + 1]
    for i in range(name_to_max_attempts[child_node_name]):
      yield self.make_move(i, child_node_name, name_to_fn, node_name_order, name_to_max_attempts)
  
  def find_random_child(self, name_to_fn: Dict[str, Callable], name_to_max_attempts: Dict[str, int], node_name_order: List[str]) -> 'Node':
    if self.terminal:
      return set()

    child_node_name = node_name_order[node_name_order.index(self.node_name) + 1]
    # for i in range(name_to_max_attempts[child_node_name]):
    return self.make_random_move(0, child_node_name, name_to_fn, node_name_order)

  def make_random_move(self, attempt_idx: int, child_node_name: str, name_to_fn: Dict[str, Callable], node_name_order: List[str]) -> 'Node':
    # 添加参数验证
    if child_node_name not in name_to_fn:
        raise KeyError(f"Function not found for node_name: {child_node_name}")
    if child_node_name not in node_name_order:
        raise ValueError(f"node_name {child_node_name} not found in node_name_order")

    fn: Callable = name_to_fn[child_node_name]
    status, child = fn(copy.deepcopy(self), random_move=True)

    if node_name_order.index(child_node_name) == len(node_name_order) - 1:
      win = True
      terminal = True
    else:
      win = False
      terminal = False
      
    child = child._asdict()
    child.update({
      'node_name': child_node_name,
      'win': win,
      'terminal': terminal,
      'attempt': attempt_idx,
      'father_hash': hash(self)
    })
    
    return self.__class__(**child)

  def is_terminal(self) -> bool:
    "Returns True if the node has no children"
    return self.terminal

  @abstractmethod
  def reward(self) -> float:
    "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
    raise NotImplementedError("Reward method must be implemented by subclasses")

  def make_move(self, attempt_idx: int, child_node_name: str, name_to_fn: Dict[str, Callable], node_name_order: List[str], name_to_max_attempts: Dict[str, int]) -> 'Node':
    # 添加参数验证
    if child_node_name not in name_to_fn:
        raise KeyError(f"Function not found for node_name: {child_node_name}")
    if child_node_name not in node_name_order:
        raise ValueError(f"node_name {child_node_name} not found in node_name_order")
    
    # next_node_name = node_name_order[node_name_order.index(self.node_name) + 1]
    fn: Callable = name_to_fn[child_node_name]
    # logger.info(f'({attempt_idx}/{name_to_max_attempts[child_node_name]}) Call {child_node_name} in [{get_source_code_name_and_line(fn)}]')
    status, child = fn(copy.deepcopy(self))

    if not status:
      win = False
      terminal = True
      logger.debug(f"Failed to expand child: {child_node_name} ({attempt_idx}/{name_to_max_attempts[child_node_name]}) in [{get_source_code_name_and_line(fn)}]")
    else:
      logger.debug(f"Successfully expand child: {child_node_name} ({attempt_idx}/{name_to_max_attempts[child_node_name]}) in [{get_source_code_name_and_line(fn)}]")
      if node_name_order.index(child_node_name) == len(node_name_order) - 1:
        win = True
        terminal = True
      else:
        win = False
        terminal = False

    
    child = child._asdict()
    child.update({
      'node_name': child_node_name,
      'win': win,
      'terminal': terminal,
      'attempt': attempt_idx,
      'father_hash': hash(self)
    })

    ret = self.__class__(**child)
    return ret

  def __str__(self) -> str:
    return json.dumps(self._asdict())
  
  def __hash__(self) -> int:
    return hash(self.__str__())

  def __eq__(self, other: Any) -> bool:
    """
    Compare two nodes for equality.
    Two nodes are considered equal if they have the same:
    - node_name
    - win status
    - terminal status
    - and all other attributes from _asdict()
    """
    if not isinstance(other, self.__class__):
      return False
    
    # Compare all attributes from _asdict()
    return self._asdict() == other._asdict()
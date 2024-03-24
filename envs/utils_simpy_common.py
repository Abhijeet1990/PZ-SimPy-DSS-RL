from abc import ABC, abstractmethod
from multiprocessing import Event
from typing import Dict,Any,Tuple
from enum import Enum
from pettingzoo import AECEnv
from gymnasium.spaces import Discrete,Box
from envs.SimComponentsDynamic import PacketGenerator, PacketSink, SwitchPort, PortMonitor, Router, Firewall,Packet
from envs.simtools import SimMan,Notifier
#from utils import seeding
import numpy as np

class Message:
    """
    A class used for the exchange of arbitrary messages between components.
    A :class:`Message` can be used to simulate both asynchronous and synchronous function
    calls.
    Attributes:
        type(Enum): An enumeration object that defines the message type
        args(Dict[str, Any]): A dictionary containing the message's arguments
        eProcessed(Event): A SimPy event that is triggered when
            :meth:`setProcessed` is called. This is useful for simulating
            synchronous function calls and also allows for return values (an
            example is provided in :meth:`setProcessed`).
    """

    def __init__(self, type: Enum, args: Dict[str, Any] = None):
        self.type = type
        self.args = args
        self.eProcessed = Event(SimMan.env)

    def setProcessed(self, returnValue: Any = None):
        """
        Makes the :attr:`eProcessed` event succeed.
        Args:
            returnValue: If specified, will be used as the `value` of the
                :attr:`eProcessed` event.
        Examples:
            If `returnValue` is specified, SimPy processes can use Signals for
            simulating synchronous function calls with return values like this:
            ::
                signal = Signal(myType, {"key", value})
                gate.output.send(signal)
                value = yield signal.eProcessed
                # value now contains the returnValue that setProcessed() was called with
        """
        self.eProcessed.succeed(returnValue)
    
    def __repr__(self):
        return "Message(type: '{}', args: {})".format(self.type.name, self.args)

class StackMessageTypes(Enum):
    """
    An enumeration of control message types to be used for the exchange of
    `Message` objects between network stack layers.
    """
    RECEIVE = 0
    SEND = 1
    ASSIGN = 2

class BaseEnv(AECEnv):
    metadata = {'render.modes': ['human'],'is_parallelizable':True}

    ASSIGNMENT_DURATION_FACTOR = 1000

    def __init__(self,  deviceCount: int, render_mode=None):
        """
        Args:
            deviceCount: The number of devices to be included in the
                environment's action space
        """
        self.deviceCount = deviceCount

        self.possible_agents = ["R1","R2","R3"]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.deviceCount = deviceCount

        self.render_mode = render_mode


class Interpreter(ABC):
    """
    An :class:`Interpreter` is an instance that observes the system's behavior
    by sniffing the packets received by the receiver and infers
    observations and rewards.
    This class serves as an abstract base class for all :class:`Interpreter`
    implementations.
    When implementing an interpreter, the following three methods have to be
    overridden:
        * :meth:`getReward`
        * :meth:`getObservation`
    The following methods provide default implementations that you might also
    want to override depending on your use case:
        * :meth:`reset`
        * :meth:`getDone`
        * :meth:`getInfo`
    """

    @abstractmethod
    def getReward(self) -> float:
        """
        Returns a reward that depends on the last channel assignment.
        """

    @abstractmethod
    def getObservation(self) -> Any:
        """
        Returns an observation of the system's state.
        """
    
    def getDone(self) -> bool:
        """
        Returns whether an episode has ended.
        Note:
            Reinforcement learning problems do not have to be split into
            episodes. In this case, you do not have to override the default
            implementation as it always returns ``False``.
        """
        return False

    def getInfo(self) -> Dict:
        """
        Returns a :class:`dict` providing additional information on the
        environment's state that may be useful for debugging but is not allowed
        to be used by a learning agent.
        """
        return {}

    def getFeedback(self) -> Tuple[Any, float, bool, Dict]:
        """
        You may want to call this at the end of a frequency band assignment to get
        feedback for your learning agent. The return values are ordered like
        they need to be returned by the :meth:`step` method of a gym
        environment.
        Returns:
            A 4-tuple with the results of :meth:`getObservation`,
            :meth:`getReward`, :meth:`getDone`, and :meth:`getInfo`
        """
        return self.getObservation(), self.getReward(), self.getDone(), self.getInfo()
    
    def reset(self):
        """
        This method is invoked when the environment is reset – override it with
        your initialization tasks if you feel like it.
        """


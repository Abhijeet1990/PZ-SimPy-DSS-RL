import functools
from queue import Queue
import random
from typing import List
import networkx as nx
import simpy
from envs.SimComponentsDynamic import Packet, PacketGenerator, SwitchPort
from envs.simtools import SimMan
from envs.utils_simpy import BaseEnv,Interpreter,Message,StackMessageTypes
from gymnasium.spaces import Box,Discrete
import numpy as np
from pettingzoo.utils import agent_selector, wrappers

def constArrival():
    #return 100000000    # time interval
    return 13.0

def constArrival2():
    #return 100000000    # time interval
    return 40.0

def constSize():
    return 25.0  # bytes

def constSize2():
    return 100.0  # bytes


NUM_ITERS = 100
adist = functools.partial(random.expovariate, 0.5)
sdist = functools.partial(random.expovariate, 0.04)  # mean size 100 bytes

adist2 = functools.partial(random.randint, 12,15)
sdist2 = functools.partial(random.expovariate, 0.0005)  # mean size 100 bytes

samp_dist = functools.partial(random.expovariate, 1.0)
port_rate = 1000.0

class SenderDevice(PacketGenerator):
    """
    """
    
    def __init__(self, env, id,  adist, sdist, initial_delay=0, finish=float("inf"), flow_id=0,debug=False,packet_size=None):
        super(SenderDevice, self).__init__(env, id,  adist, sdist)
        self.initial_delay=initial_delay
        self.out_channel = None
        self.debug=debug
        self.packet_size = packet_size
        SimMan.process(self.senderProcess())
    
    def senderProcess(self):
        yield SimMan.timeout(self.initial_delay)
        while SimMan.now < self.finish:
            # wait for next transmission
            yield SimMan.timeout(self.adist())
            self.packets_sent += 1

            p = Packet(SimMan.now, self.sdist(), self.packets_sent, src=self.id, flow_id=self.flow_id)
            if self.packet_size is not None:
                p.size = float(self.packet_size)

            #self.out.put(p)
            self.out_channel.put(p)

            if self.debug:
                print('gen '+str(self.env.now)+' '+str(p))

    def sendCommand(self):
        yield SimMan.timeout(self.adist())
        self.packets_sent +=1
        p = Packet(SimMan.now, self.sdist(), self.packets_sent, src=self.id, flow_id=self.flow_id)
        if self.debug:
            print(str(self.id) + ' : Sending control command')
        self.out.put(p)

class ForwarderDevice(SwitchPort):
    def __init__(self, env,id, rate, qlimit=None, limit_bytes=True, debug=False):
        super(ForwarderDevice, self).__init__(env,id, rate, qlimit, limit_bytes, debug)
        self.debug = debug
        self.out_channel = []
        self.selected_Channel_Index = 0
        self.queue_update_freq=25
        SimMan.process(self.forwarderProcess())
        SimMan.process(self.update_router_queue())
    
    def forwarderProcess(self):
        while True:
            msg = (yield self.store.get())
            self.busy = 1
            self.byte_size -= msg.size
            # take some time for the router to process the packet
            yield SimMan.timeout(msg.size*8.0/(self.rate))
            try:
                # here the out should be the channel instead of the router
                self.out_channel[self.selected_Channel_Index].put(msg)
            except:
                print(self.id)
            self.busy = 0
            if self.debug:
                print('No of outchannels {0} their ids {1}'.format(len(self.out_channel),[x.cid for x in self.out_channel]))
                print('At '+str(self.env.now)+', '+str(self.id)+':'+str(self.env.now)+' : ' +str(msg))
                print(str(self.env.now)+' : Selected Channel of Router {0} is {1}'.format(self.id, self.selected_Channel_Index))

    def update_router_queue(self):
        while True:
            if self.debug:
                print('At {0}, {1} percent filled : {2}'.format(self.env.now, self.id,self.temp_byte_size/self.qlimit))
            self.temp_byte_size = 0
            yield SimMan.timeout(self.queue_update_freq)

    # this function will change the receiver if it founds one receiver to be busy
    def change_receiver(self, new_receiver):
        self.selected_Channel_Index = new_receiver
        if self.debug:
            print(str(self.env.now)+' : '+str(self.id) + ' : Changing Route to ' + str(self.out[self.selected_Channel_Index].id))
        yield SimMan.timeout(1)

class ReceiverDevice(Interpreter):
    def __init__(self, simpyenv,id, _routers,_channels,_senders, rec_arrivals=False, absolute_arrivals=False, rec_waits=True, debug=False, selector=None):
        self.store = simpy.Store(simpyenv)
        self.env = simpyenv
        #self.gymenv = gymenv
        self.id = id
        self.rec_waits = rec_waits
        self.rec_arrivals = rec_arrivals
        self.absolute_arrivals = absolute_arrivals
        self.waits = []
        self.arrivals = []
        self.debug = debug
        self.packets_rec = 0
        self.packets_record ={}
        self.bytes_rec = 0
        self.selector = selector
        self.last_arrival = 0.0
        self.routers = _routers
        self.channels = _channels
        self.senders = _senders
        self.reset()
        

    def put(self, pkt):
        if not self.selector or self.selector(pkt):
            now = self.env.now

            if self.rec_waits and pkt.src in ['PG1','PG2']:# small case
                self.waits.append(self.env.now - pkt.time)
           
            if self.rec_arrivals:
                if self.absolute_arrivals:
                    self.arrivals.append(now)
                else:
                    self.arrivals.append(now - self.last_arrival)
                self.last_arrival = now

            if pkt.src in ['PG1','PG2']:
                self.packets_rec += 1
                if pkt.src in self.packets_record.keys():
                    self.packets_record[pkt.src]+=1
                else:
                    self.packets_record[pkt.src]=1

            self.bytes_rec += pkt.size
            if self.debug:
                print('At '+str(self.env.now)+', '+str(self.id)+':'+str(self.env.now)+' '+str(pkt))
    
    def reset(self):
        self.receivedPackets = [0 for _ in range(len(self.senders))]
        self._done = False

    def getReward(self):
        """
        In the cyber deception environment, reward would be based on how easily an attacker is attracted
        to the honeypot instead of original DNP3 node.
        """
        try:
            rewards = []
            if self.id == "PS":
                gthan10 = dict((k, v) for k, v in self.packets_record.items() if v >= 10)
                if len(gthan10) ==1:
                    reward = 2
                elif len(gthan10) ==2:
                    reward = 5
                else:
                    reward = -2
            for rtr in self.routers:
                rewards.append((1-float(rtr.packets_drop)/float(rtr.packets_rec))*reward)
            return rewards
        except:
            return [0.0 for i in range(len(self.routers))]

    def getObservation(self):
        observations = []
        for router in self.routers:
            observations.append(router.packets_drop)
        
        # we add the channel utilization rate to this also and construct the weighted graph for shortest path
        for channel in self.channels:
            observations.append(channel.utilization_rate)

        return np.array(observations)
    
    def getDone(self):
        if self.id == "PS":
            gthan10 = dict((k, v) for k, v in self.packets_record.items() if v >= 10)
            if len(gthan10) == 2 or self.env.now > 500.0:
                self._done = True
            
        return self._done
    
    def getInfo(self):
        # DQN in keras-rl crashes when the values are iterable, thus the
        # string below
        return {"Last arrived packet": str(self.last_arrival)}


class Channel():
    def __init__(self,env,id, src,dest, bw=2000, delay=1, limit_bytes=True,debug = False, snr = 10):
        self.store = simpy.Store(env)
        self.cid = id
        self.packets_rec = 0
        self.packets_drop = 0
        self.env = env # simpy env
        self.src = src # source node
        self.dest = dest # destination node
        self.bw = bw # channel bandwidth
        self.byte_size = 0  # Number of data size already in the channel
        self.delay = delay # channel delay (this should be propagation delay)
        self.temp_byte_size = 0
        self.limit_bytes = limit_bytes
        self.debug = debug
        #self.channel_capacity = self.bw * math.log10(1 + snr) # shannon's channel capacity formula for noisy channel
        # for noiseless channel
        # the utilization rate is usually computed in a time interval how much bytes are served
        self.utilization_rate = 0
        self.channel_capacity = self.bw
        self.ur_update_freq=25
        SimMan.process(self.run())
        SimMan.process(self.update_ur())

    def run(self):
        while True:
            msg = (yield self.store.get())
            # this first expression is transmission delay and second the propagation
            try:
                latency =  msg.size*8.0/self.channel_capacity  + self.delay
            except:
                latency =  msg.size*8.0/(self.channel_capacity+100)  + self.delay
            if latency < 0:
                latency = 0
            yield SimMan.timeout(latency) 
            """ try:
                yield SimMan.timeout(latency) 
            except:
                print(self.channel_capacity) """
            self.dest.put(msg)
            if self.debug:
                print(msg)

    def put(self, pkt):
        self.packets_rec += 1
        tmp_byte_count = self.byte_size + pkt.size
        self.channel_capacity = self.bw * (1 - self.utilization_rate)
        #print('{0} I am called by router'.format(self.cid))
        if self.debug:
            print('{0}: testing channel capacity {1} and tmp_byte_size {2}'.format(self.env.now, self.channel_capacity, self.temp_byte_size))
            print('{0}: utilization rate: {1}'.format(self.cid, self.utilization_rate))
        if self.channel_capacity is None:
            self.byte_size = tmp_byte_count
            self.temp_byte_size = self.byte_size
            return self.store.put(pkt)
        if self.limit_bytes and self.temp_byte_size >= self.channel_capacity:
            self.packets_drop += 1
            return
        elif not self.limit_bytes and len(self.store.items) >= self.channel_capacity-1:
            self.packets_drop += 1
        else:
            self.byte_size = tmp_byte_count
            self.temp_byte_size+= self.byte_size
            return self.store.put(pkt)

    # schedule every 10 sec to update the Utilization Rate
    def update_ur(self):
        while True:
            self.utilization_rate = self.temp_byte_size/(self.ur_update_freq*self.bw)
            if self.debug:
                print('At {0}, {1} update  utilization val : {2}'.format(self.env.now, self.cid,str(self.utilization_rate)))
            self.temp_byte_size = 0
            yield SimMan.timeout(self.ur_update_freq)

class WiredChannel(Channel):
    def __init__(self,env, src,dest):
        super(CyberEnv.WiredChannel,self).__init__(env,src,dest)
        SimMan.process(self.run())

    def run(self):
        NotImplementedError

class WirelessChannel(Channel):
    def __init__(self,env, src,dest):
        super(CyberEnv.WirelessChannel,self).__init__(env,src,dest)
        SimMan.process(self.run())

    def run(self):
        NotImplementedError



def reinitialize_midsize_network():
        SimMan.init()
        senders: List[SenderDevice] = [
            SenderDevice(SimMan.env, 'PG1', constArrival, sdist ,debug=False),
            SenderDevice(SimMan.env, 'PG2', constArrival2, sdist,debug=False)
        ]

        # initialize all the forwarding devices
        routers: List[ForwarderDevice] = [
            ForwarderDevice(SimMan.env, 'R1',rate=200.0, qlimit=200),
            ForwarderDevice(SimMan.env,'R2',rate=200.0, qlimit=200),
            ForwarderDevice(SimMan.env,'R3',rate=200.0, qlimit=200*random.randint(0,1)),
            ForwarderDevice(SimMan.env,'R4',rate=200.0, qlimit=200*random.randint(0,1)),
            ForwarderDevice(SimMan.env,'R5',rate=200.0, qlimit=200)
        ]
        
        
        
        channels : List[Channel] = [
            Channel(SimMan.env, 'CG1',src = senders[0],dest=routers[0]),
            Channel(SimMan.env, 'CG2',src = senders[1],dest=routers[1]),
            Channel(SimMan.env, 'C12',src = routers[0],dest=routers[1]),
            Channel(SimMan.env, 'C13',src = routers[0],dest=routers[2]),
            Channel(SimMan.env, 'C14',src = routers[0],dest=routers[3]),
            Channel(SimMan.env, 'C23',src = routers[1],dest=routers[2]),
            Channel(SimMan.env, 'C24',src = routers[1],dest=routers[3]),
            Channel(SimMan.env, 'C35',src = routers[2],dest=routers[4]),
            Channel(SimMan.env, 'C45',src = routers[3],dest=routers[4])
            #,
            #Channel(SimMan.env, 'CS',src = routers[4],dest=interpreter)
        ]

        # actual recepient node
        interpreter = ReceiverDevice(SimMan.env, 'PS',_routers=routers,_senders=senders,_channels=channels)

        # create the network, i.e. connect the edges
        senders[0].out = routers[0]
        senders[1].out = routers[1]
        routers[0].out.append(routers[1])
        routers[0].out.append(routers[2])
        routers[0].out.append(routers[3])
        routers[1].out.append(routers[2])
        routers[1].out.append(routers[3])
        routers[2].out.append(routers[4])
        routers[3].out.append(routers[4])
        routers[4].out.append(interpreter)

        channels.append(Channel(SimMan.env, 'CS',src = routers[4],dest=interpreter))

        # initialize the channels
        senders[0].out_channel = channels[0]
        senders[1].out_channel = channels[1]
        routers[0].out_channel.append(channels[2])
        routers[0].out_channel.append(channels[3])
        routers[0].out_channel.append(channels[4])
        routers[1].out_channel.append(channels[5])
        routers[1].out_channel.append(channels[6])
        routers[2].out_channel.append(channels[7])
        routers[3].out_channel.append(channels[8])
        routers[4].out_channel.append(channels[9])

        return routers,senders,interpreter,channels

_routers,_senders,_interpreter,_channels = reinitialize_midsize_network()

def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = CyberEnv(render_mode=internal_render_mode,envDebug=False,with_threat=True)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    #env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class CyberEnv(BaseEnv):
    
    def __init__(self, channelModel=True, envDebug=True, R2_qlimit = 100, ch_bw=1000, with_threat= False, comp_zones = None,render_mode=None):
        super(CyberEnv, self).__init__(deviceCount=5)
        global _channels
        self.channelModel = channelModel
        self.envDebug = envDebug
        self.R2_qlimit = R2_qlimit
        self.ch_bw = ch_bw
        self.channel_map ={}
        self.with_threat = with_threat
        self.comp_zones = comp_zones
        self.small_Case = False
        # if provided_graph is None:
        #     self.small_Case = True
        #     # Here we monitor the number of lost packets at all the router nodes (which is kind of the device count)
        #     self.G = nx.Graph()
        #     self.reinitialize_midsize_network()
        #self.observation_space = Box(low=0, high=100.0, shape=(len(_channels)+1,), dtype=np.float32)
            # if channelModel is True:
            #     self.observation_space = Box(low=0, high=1000000.0, shape=(len(self.channels)+1,), dtype=np.float32)
            # self.action_space = Discrete(3)

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    #@functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        global _channels
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Box(low=0, high=100.0, shape=(len(_channels)+1,), dtype=np.float32)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    #@functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        global _routers
        ix = self.possible_agents.index(agent)
        return Discrete(len(_routers[ix].out_channel))

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])
        
    def reset(self,seed=None, options=None):
        """
        Resets the state of the environment and returns an initial observation.
        """
        global _routers,_senders,_interpreter,_channels
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: self.observation_space(agent).sample() for agent in self.agents}
        self.num_moves = 0

        _routers,_senders,_interpreter,_channels = reinitialize_midsize_network()
      
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        
    def step(self, action):
        global _routers,_interpreter,_senders
        #print(f"In step for Agent {self.agent_selection}")

        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.state[self.agent_selection] = action

        ix = self.possible_agents.index(agent)

        SimMan.process(_routers[ix].change_receiver(action))

        # only move forward once
        if ix == 0:
            SimMan.runSimulation(50)

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            for ix,agent in enumerate(self.agents):
                obs_all, rws,self.terminations[agent],self.infos[agent] = _interpreter.getFeedback()
                self.rewards[agent] = rws[ix]
                # obtain the router packet drop rate
                self.observations[agent] = [obs_all[ix]]
                # get the channel utilization rate for others
                self.observations[agent].extend(obs_all[len(self.agents):])

            self.num_moves += 1
            # The truncations dictionary must be updated for all players.
            self.truncations = {
                agent: self.num_moves >= NUM_ITERS for agent in self.agents
            }

        else:
            # necessary so that observe() returns a reasonable observation at all times.
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = 0
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        # for k,v in self.rewards.items():
        #     print(f"Reward for Agent {k} = {v}")
        #     print(f"Obs for Agent {k} = {self.observations[k]}")


        if self.render_mode == "human":
            self.render()


    def render(self, mode='human', close=False):
        if len(self.agents) == 5:
            string = "Current state of all router"
        else:
            string = "Game over"
        print(string)
        pass
        
    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])
    
    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass
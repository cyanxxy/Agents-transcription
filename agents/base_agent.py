"""Base Agent Framework for ExactTranscriber"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
from datetime import datetime
import uuid


class MessageType(Enum):
    """Types of messages agents can send"""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    STATUS = "status"
    BROADCAST = "broadcast"


class AgentStatus(Enum):
    """Agent lifecycle states"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class Message:
    """Message structure for inter-agent communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    recipient: str = ""
    type: MessageType = MessageType.REQUEST
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    
    def reply(self, content: Dict[str, Any], type: MessageType = MessageType.RESPONSE) -> 'Message':
        """Create a reply to this message"""
        return Message(
            sender=self.recipient,
            recipient=self.sender,
            type=type,
            content=content,
            correlation_id=self.id
        )


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str, supervisor: Optional['AgentSupervisor'] = None):
        self.name = name
        self.supervisor = supervisor
        self.status = AgentStatus.IDLE
        self.logger = logging.getLogger(f"Agent.{name}")
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._capabilities: List[str] = []
        
    @abstractmethod
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process incoming message and return response if needed"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this agent provides"""
        pass
    
    async def start(self):
        """Start the agent's message processing loop"""
        self._running = True
        self.status = AgentStatus.IDLE
        self.logger.info(f"{self.name} agent started")
        
        while self._running:
            try:
                # Wait for message with timeout to allow shutdown checks
                message = await asyncio.wait_for(
                    self._message_queue.get(), 
                    timeout=1.0
                )
                
                self.status = AgentStatus.BUSY
                self.logger.debug(f"Processing message: {message.id}")
                
                # Process the message
                response = await self.process_message(message)
                
                # Send response if generated
                if response and self.supervisor:
                    await self.supervisor.route_message(response)
                
                self.status = AgentStatus.IDLE
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                self.status = AgentStatus.ERROR
                
                # Send error response
                if message and self.supervisor:
                    error_response = message.reply(
                        {"error": str(e)},
                        MessageType.ERROR
                    )
                    await self.supervisor.route_message(error_response)
    
    async def stop(self):
        """Stop the agent"""
        self._running = False
        self.status = AgentStatus.SHUTDOWN
        self.logger.info(f"{self.name} agent stopped")
    
    async def send_message(self, message: Message):
        """Add message to processing queue"""
        await self._message_queue.put(message)
    
    def can_handle(self, capability: str) -> bool:
        """Check if agent can handle a specific capability"""
        return capability in self.get_capabilities()


class AgentSupervisor:
    """Manages and coordinates multiple agents"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.logger = logging.getLogger("AgentSupervisor")
        self._running = False
        
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the supervisor"""
        self.agents[agent.name] = agent
        agent.supervisor = self
        self.logger.info(f"Registered agent: {agent.name}")
    
    async def start_all(self):
        """Start all registered agents"""
        self._running = True
        tasks = []
        
        for agent in self.agents.values():
            task = asyncio.create_task(agent.start())
            tasks.append(task)
        
        self.logger.info("All agents started")
        
        # Keep tasks running
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Error in agent tasks: {e}")
    
    async def stop_all(self):
        """Stop all agents"""
        self._running = False
        
        for agent in self.agents.values():
            await agent.stop()
        
        self.logger.info("All agents stopped")
    
    async def route_message(self, message: Message):
        """Route message to appropriate agent"""
        if message.recipient in self.agents:
            await self.agents[message.recipient].send_message(message)
        elif message.type == MessageType.BROADCAST:
            # Send to all agents except sender
            for name, agent in self.agents.items():
                if name != message.sender:
                    await agent.send_message(message)
        else:
            self.logger.warning(f"Unknown recipient: {message.recipient}")
    
    async def request(self, 
                     sender: str, 
                     recipient: str, 
                     content: Dict[str, Any],
                     timeout: float = 30.0) -> Optional[Message]:
        """Send request and wait for response"""
        request = Message(
            sender=sender,
            recipient=recipient,
            type=MessageType.REQUEST,
            content=content
        )
        
        # Create future for response
        response_future = asyncio.Future()
        
        # Store future with correlation ID
        # (In production, implement proper response tracking)
        
        await self.route_message(request)
        
        try:
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            self.logger.error(f"Request timeout: {request.id}")
            return None
    
    def find_agent_for_capability(self, capability: str) -> Optional[BaseAgent]:
        """Find agent that can handle specific capability"""
        for agent in self.agents.values():
            if agent.can_handle(capability):
                return agent
        return None
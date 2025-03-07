from mlagents_envs.side_channel import SideChannel, IncomingMessage, OutgoingMessage
from mlagents_envs.exception import (
    UnityCommunicationException,
    UnitySideChannelException,
)

import uuid
from typing import Optional


class AgentResetChannel(SideChannel):
    def __init__(self) -> None:
        super().__init__(uuid.UUID("e951342c-4f7e-11ea-b238-784f43874323"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Is called by the environment to the side channel. Can be called
        multiple times per step if multiple messages are meant for that
        SideChannel.
        Note that Python should never receive an agent reset from Unity
        """
        raise UnityCommunicationException(
            "The AgentConfigurationChannel received a message from Unity, "
            + "this should not have happened."
        )

    def set_reset_pose(self, random_reset: bool = False, x: Optional[float] = None, y: Optional[float] = None,
                       z: Optional[float] = None, yaw: Optional[float] = None) -> None:
        """
        Send agent reset pose to Unity (in Unity coordinate frame).
        Only agent yaw can be reset in rotation.
        :param random_reset: randomly reset camera pose, all subsequent arguments will be ignored
        :param x: agent reset x in meters
        :param y: agent reset y in meters
        :param z: agent reset z in meters
        :param yaw: agent reset yaw in degrees
        :return: None
        """
        if random_reset:
            msg = OutgoingMessage()
            msg.write_bool(random_reset)
            super().queue_message_to_send(msg)
            return

        all_none: bool = x is None and y is None and z is None and yaw is None
        assert not all_none, 'Requested external reset but pose are all none!'

        assert x is not None and y is not None and z is not None and yaw is not None, \
            f'Agent reset pose should not have any none component!'

        msg = OutgoingMessage()
        msg.write_bool(random_reset)
        msg.write_float32(x)
        msg.write_float32(y)
        msg.write_float32(z)
        msg.write_float32(yaw)
        super().queue_message_to_send(msg)


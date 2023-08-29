from mlagents_envs.side_channel.raw_bytes_channel import RawBytesChannel
from mlagents_envs.exception import (
    UnityCommunicationException,
    UnitySideChannelException,
)
import uuid


class SegmentationReceiverChannel(RawBytesChannel):
    def __init__(self) -> None:
        super().__init__(uuid.UUID("a1d8f7b7-cec8-50f9-b78b-d3e165a70987"))

    def get_segmentation_mask(self):
        pixel_bytes_list = super().get_and_clear_received_messages()
        if len(pixel_bytes_list) == 0:
            # print(f'Mask bytes list is empty!')
            return None

        pixel_bytes = pixel_bytes_list[-1]  # get the latest mask

        # print(f'Mask bytes len is {len(list(pixel_bytes))}')
        # print(f'{pixel_bytes=}')
        # print(f'{list(pixel_bytes)=}')

        return pixel_bytes


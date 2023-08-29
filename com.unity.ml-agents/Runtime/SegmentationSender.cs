using System;
using System.Linq;
using Unity.MLAgents.SideChannels;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;


namespace Unity.MLAgents
{
    public sealed class SegmentationSender
    {

        const string k_SegmentationChannelDefaultId = "a1d8f7b7-cec8-50f9-b78b-d3e165a70987";

        private readonly RawBytesChannel m_Channel;

        private readonly PerceptionCamera m_PerceptionCamera;

        internal SegmentationSender()
        {
            // get main camera gameobject reference
            var mainCamera = GameObject.FindGameObjectWithTag("MainCamera");
            if (!mainCamera)
            {
                Debug.Log("Object MainCamera is not found!");
                return;
            }

            // get perceptioncamera component reference
            m_PerceptionCamera = mainCamera.GetComponent<PerceptionCamera>();
            if (!m_PerceptionCamera)
            {
                Debug.Log("Component PerceptionCamera is not found!");
                return;
            }

            // init rawbytes side channel and register it
            m_Channel = new RawBytesChannel(new Guid(k_SegmentationChannelDefaultId));
            SideChannelManager.RegisterSideChannel(m_Channel);
        }

        // get segmentation image and send using m_Channel.SendRawBytes
        public void SendSegmentationImage()
        {
            var segBytes = m_PerceptionCamera.maskBytes;

            if (segBytes.Length == 0)
            {
                // Debug.Log("Mask bytes empty!");
            }
            else
            {
                // Debug.Log("Send mask bytes with len " + segBytes.Length);
                m_Channel.SendRawBytes(segBytes);
            }
        }

        internal void Dispose()
        {
            // SideChannelManager.UnregisterSideChannel(m_Channel);
        }

    }
}


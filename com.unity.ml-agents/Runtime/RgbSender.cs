using System;
using System.Linq;
using Unity.MLAgents.SideChannels;
using UnityEngine;
// using UnityEngine.Perception.GroundTruth;


namespace Unity.MLAgents
{
    public sealed class RGBSender
    {

        const string k_RGBChannelDefaultId = "a1d8f7b7-cec8-50f9-b78b-d3e165a71234";

        private readonly RawBytesChannel m_Channel;

        // private readonly PerceptionCamera m_PerceptionCamera;

        internal RGBSender()
        {
            // get main camera gameobject reference
            var mainCamera = GameObject.FindGameObjectWithTag("MainCamera");
            if (!mainCamera)
            {
                Debug.Log("Object MainCamera is not found!");
                return;
            }

            // get perceptioncamera component reference
            // m_PerceptionCamera = mainCamera.GetComponent<PerceptionCamera>();
            // if (!m_PerceptionCamera)
            // {
            //     Debug.Log("Component PerceptionCamera is not found!");
            //     return;
            // }

            // init rawbytes side channel and register it
            m_Channel = new RawBytesChannel(new Guid(k_RGBChannelDefaultId));
            SideChannelManager.RegisterSideChannel(m_Channel);
        }

        // get segmentation image and send using m_Channel.SendRawBytes
        public void SendRGBImage()
        {
            // var rgbBytes = m_PerceptionCamera.rgbBytes;
            //
            // if (rgbBytes.Length == 0)
            // {
            //     // Debug.Log("rgb bytes empty!");
            // }
            // else
            // {
            //     // Debug.Log("Send rgb bytes with len " + rgbBytes.Length);
            //     m_Channel.SendRawBytes(rgbBytes);
            // }
        }

        internal void Dispose()
        {
            // SideChannelManager.UnregisterSideChannel(m_Channel);
        }

    }
}



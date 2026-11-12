using UnityEngine;
using UnityEngine.XR;

public class DESIVRNavigation : MonoBehaviour
{
    [Header("XR Origin")]
    public Transform xrOrigin;
    public Transform headCamera;

    [Header("Movement")]
    public float movementSpeed = 5f;
    public float rotationSpeed = 70f;

    [Header("Astronomical navigation")]
    public float fastMultiplier = 4f;

    private InputDevice leftController;
    private InputDevice rightController;

    private void Update()
    {
        if (!leftController.isValid)
        {
            leftController =
                InputDevices.GetDeviceAtXRNode(
                    XRNode.LeftHand
                );
        }

        if (!rightController.isValid)
        {
            rightController =
                InputDevices.GetDeviceAtXRNode(
                    XRNode.RightHand
                );
        }

        if (xrOrigin == null || headCamera == null)
            return;

        leftController.TryGetFeatureValue(
            CommonUsages.primary2DAxis,
            out Vector2 moveInput
        );

        rightController.TryGetFeatureValue(
            CommonUsages.primary2DAxis,
            out Vector2 turnInput
        );

        leftController.TryGetFeatureValue(
            CommonUsages.primaryButton,
            out bool fastMode
        );

        float speed = movementSpeed;

        if (fastMode)
            speed *= fastMultiplier;

        // Move relative to headset direction.
        Vector3 forward = headCamera.forward;
        Vector3 right = headCamera.right;

        Vector3 movement =
            forward * moveInput.y +
            right * moveInput.x;

        xrOrigin.position +=
            movement * speed * Time.deltaTime;

        // Rotate around the headset position.
        float yaw =
            turnInput.x * rotationSpeed * Time.deltaTime;

        Vector3 pivot = headCamera.position;

        xrOrigin.RotateAround(
            pivot,
            Vector3.up,
            yaw
        );
    }
}

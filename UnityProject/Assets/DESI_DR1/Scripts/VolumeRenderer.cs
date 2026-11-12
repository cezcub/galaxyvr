
using UnityEngine;

public class DESIVolumeRenderer : MonoBehaviour
{
    [Header("References")]
    public DESIDataLoader dataLoader;
    public Shader volumeShader;

    [Header("Astronomical dimensions (Mpc)")]
    public Vector3 gridMin = new Vector3(
        -4708.65363f,
        -4797.25261f,
        -3576.17891f
    );

    public Vector3 gridMax = new Vector3(
        4708.32480f,
        4619.72582f,
        5840.79951f
    );

    [Header("Display scale")]
    [Tooltip("Unity units per Mpc")]
    public float unityUnitsPerMpc = 0.01f;

    [Header("Rendering")]
    [Range(0, 2)]
    public int visualizationMode = 2;

    [Range(16, 128)]
    public int raySteps = 64;

    [Range(0.01f, 2f)]
    public float opacity = 0.35f;

    [Header("Density transfer function")]
    public float densityMin = 0f;
    public float densityMax = 3f;

    [Range(0f, 1f)]
    public float densityThreshold = 0.02f;

    [Range(0.1f, 5f)]
    public float brightness = 1.5f;

    private Material volumeMaterial;
    private GameObject volumeCube;

    private void Start()
    {
        Initialize();
    }

    public void Initialize()
    {
        if (dataLoader == null)
        {
            Debug.LogError("Assign DESIDataLoader.");
            return;
        }

        if (volumeShader == null)
        {
            Debug.LogError("Assign DESI volume shader.");
            return;
        }

        if (!dataLoader.IsLoaded)
        {
            dataLoader.LoadData();
        }

        CreateVolume();
        UpdateShaderSettings();
    }

    private void CreateVolume()
    {
        Vector3 dimensionsMpc = gridMax - gridMin;

        Vector3 centerMpc = (gridMin + gridMax) * 0.5f;

        Vector3 dimensionsUnity =
            dimensionsMpc * unityUnitsPerMpc;

        // The texture's UV space represents the
        // complete astronomical bounding volume.
        volumeCube = GameObject.CreatePrimitive(
            PrimitiveType.Cube
        );

        volumeCube.name = "DESI Cosmic Web Volume";

        volumeCube.transform.SetParent(transform, false);
        volumeCube.transform.localPosition = Vector3.zero;
        volumeCube.transform.localRotation = Quaternion.identity;
        volumeCube.transform.localScale = dimensionsUnity;

        // No collider needed for volume rendering.
        Collider cubeCollider =
            volumeCube.GetComponent<Collider>();

        if (cubeCollider != null)
            Destroy(cubeCollider);

        volumeMaterial = new Material(volumeShader);

        volumeMaterial.SetTexture(
            "_StructureTex",
            dataLoader.StructureTexture
        );

        volumeMaterial.SetTexture(
            "_DensityTex",
            dataLoader.DensityTexture
        );

        volumeMaterial.SetVector(
            "_VolumeCenterMpc",
            centerMpc
        );

        volumeMaterial.SetVector(
            "_VolumeDimensionsMpc",
            dimensionsMpc
        );

        volumeCube.GetComponent<MeshRenderer>().material =
            volumeMaterial;

        volumeCube.SetActive(volumeVisible);

        Debug.Log(
            $"DESI universe created. Dimensions: {dimensionsUnity}"
        );

        Debug.Log(
            $"Astronomical center (Mpc): {centerMpc}"
        );
    }

    public void UpdateShaderSettings()
    {
        if (volumeMaterial == null)
            return;

        volumeMaterial.SetInt(
            "_VisualizationMode",
            visualizationMode
        );

        volumeMaterial.SetInt(
            "_RaySteps",
            raySteps
        );

        volumeMaterial.SetFloat(
            "_Opacity",
            opacity
        );

        volumeMaterial.SetFloat(
            "_DensityMin",
            densityMin
        );

        volumeMaterial.SetFloat(
            "_DensityMax",
            Mathf.Max(densityMax, densityMin + 0.0001f)
        );

        volumeMaterial.SetFloat(
            "_DensityThreshold",
            densityThreshold
        );

        volumeMaterial.SetFloat(
            "_Brightness",
            brightness
        );
    }

    public void SetVisualizationMode(int mode)
    {
        visualizationMode = Mathf.Clamp(mode, 0, 2);
        UpdateShaderSettings();
    }

    public void SetOpacity(float value)
    {
        opacity = Mathf.Clamp(value, 0.01f, 2f);
        UpdateShaderSettings();
    }

    public void SetDensityThreshold(float value)
    {
        densityThreshold = Mathf.Clamp01(value);
        UpdateShaderSettings();
    }

    public void SetRaySteps(int value)
    {
        raySteps = Mathf.Clamp(value, 16, 128);
        UpdateShaderSettings();
    }

    public void SetBrightness(float value)
    {
        brightness = Mathf.Clamp(value, 0.1f, 5f);
        UpdateShaderSettings();
    }

    private void OnDestroy()
    {
        if (volumeMaterial != null)
            Destroy(volumeMaterial);

        if (volumeCube != null)
            Destroy(volumeCube);
    }

    private bool volumeVisible = true;

    public void SetVisible(bool visible)
    {
        volumeVisible = visible;

        if (volumeCube != null)
            volumeCube.SetActive(visible);
    }

}

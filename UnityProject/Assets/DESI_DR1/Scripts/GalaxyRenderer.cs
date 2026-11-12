using UnityEngine;
using UnityEngine.Networking;
using System;
using System.Collections;
using System.IO;

public class DESIGalaxyRenderer : MonoBehaviour
{
    [Header("Binary data")]
    public string galaxyFile = "galaxies.bin";

    public int headerSize = 12;
    public int recordSize = 24;

    [Header("Coordinate origin (Mpc)")]
    public Vector3 originMpc = new Vector3(
        -0.16441595f,
        -88.76339571f,
        1132.31030129f
    );

    [Header("Scale")]
    public float unityUnitsPerMpc = 0.01f;

    [Header("Rendering")]
    public Material galaxyMaterial;

    public float galaxySize = 0.025f;
    public float brightness = 1.5f;

    private GraphicsBuffer galaxyBuffer;

    private int galaxyCount;

    private bool isReady;
    private bool visible = true;

    public bool IsReady => isReady;
    public int GalaxyCount => galaxyCount;

    public void SetVisible(bool value)
    {
        visible = value;
    }

    private void Start()
    {
        StartCoroutine(LoadGalaxies());
    }

    private IEnumerator LoadGalaxies()
    {
        string path = Path.Combine(
            Application.streamingAssetsPath,
            "DESI_DR1",
            galaxyFile
        );

        using (UnityWebRequest request =
            UnityWebRequest.Get(path))
        {
            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError(
                    "Galaxy loading failed: " + request.error
                );
                yield break;
            }

            byte[] bytes = request.downloadHandler.data;

            if (bytes.Length < headerSize ||
                (bytes.Length - headerSize) % recordSize != 0)
            {
                Debug.LogError(
                    "Invalid galaxy binary file size."
                );
                yield break;
            }

            if (!BitConverter.IsLittleEndian)
            {
                Debug.LogError(
                    "This loader expects a little-endian platform."
                );
                yield break;
            }

            int totalGalaxies =
                (bytes.Length - headerSize) / recordSize;

            galaxyCount = totalGalaxies;

            Vector4[] positions =
                new Vector4[galaxyCount];

            Vector3[] coordinates =
                new Vector3[galaxyCount];

            float[] redshifts =
                new float[galaxyCount];

            ulong[] targetIDs =
                new ulong[galaxyCount];

            for (int i = 0; i < galaxyCount; i++)
            {
                int offset =
                    headerSize + i * recordSize;

                float x = BitConverter.ToSingle(
                    bytes, offset
                );

                float y = BitConverter.ToSingle(
                    bytes, offset + 4
                );

                float z = BitConverter.ToSingle(
                    bytes, offset + 8
                );

                float redshift = BitConverter.ToSingle(
                    bytes, offset + 12
                );

                ulong targetID = BitConverter.ToUInt64(
                    bytes, offset + 16
                );

                Vector3 positionMpc = new Vector3(
                    x, y, z
                );

                Vector3 unityPosition =
                    (positionMpc - originMpc) *
                    unityUnitsPerMpc;

                Vector3 worldPosition =
                    transform.TransformPoint(localPosition);
                
                positions[i] = new Vector4(
                    unityPosition.x,
                    unityPosition.y,
                    unityPosition.z,
                    1f
                );

                coordinates[i] = unityPosition;
                redshifts[i] = redshift;
                targetIDs[i] = targetID;
            }

            galaxyBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                galaxyCount,
                16
            );

            galaxyBuffer.SetData(positions);

            if (galaxyMaterial == null)
            {
                Debug.LogError(
                    "Assign galaxy material."
                );
                yield break;
            }

            galaxyMaterial.SetBuffer(
                "_GalaxyPositions",
                galaxyBuffer
            );

            galaxyMaterial.SetFloat(
                "_GalaxySize",
                galaxySize
            );

            galaxyMaterial.SetFloat(
                "_Brightness",
                brightness
            );

            // Retain metadata for future interaction.
            GalaxyCoordinates = coordinates;
            GalaxyRedshifts = redshifts;
            GalaxyTargetIDs = targetIDs;

            isReady = true;

            Debug.Log(
                $"Loaded {galaxyCount:N0} DESI galaxies."
            );

            Debug.Log(
                $"Binary file size: {bytes.Length:N0} bytes"
            );

            bytes = null;
            positions = null;
        }
    }

    public Vector3[] GalaxyCoordinates { get; private set; }

    public float[] GalaxyRedshifts { get; private set; }

    public ulong[] GalaxyTargetIDs { get; private set; }

    private void LateUpdate()
    {
        if (!isReady || !visible ||
            galaxyMaterial == null)
            return;

        Vector3 center =
            transform.TransformPoint(Vector3.zero);

        RenderParams renderParams =
            new RenderParams(galaxyMaterial);

        renderParams.worldBounds = new Bounds(
            center,
            Vector3.one * 100f
        );

        renderParams.shadowCastingMode =
            UnityEngine.Rendering.ShadowCastingMode.Off;

        renderParams.receiveShadows = false;

        Graphics.RenderPrimitives(
            ref renderParams,
            MeshTopology.Triangles,
            6,
            galaxyCount
        );
    }

    private void OnDestroy()
    {
        if (galaxyBuffer != null)
        {
            galaxyBuffer.Release();
            galaxyBuffer = null;
        }
    }

    private bool visible = true;

    public void SetVisible(bool value)
    {
        visible = value;
    }

}
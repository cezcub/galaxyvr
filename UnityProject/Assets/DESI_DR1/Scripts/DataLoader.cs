using UnityEngine;
using System;
using System.IO;

public class DESIDataLoader : MonoBehaviour
{
    [Header("Dataset")]
    public int resolution = 256;
    public string structuresFile = "structures.bin";
    public string densityFile = "density.bin";

    [Header("Textures")]
    public FilterMode densityFilter = FilterMode.Trilinear;

    public Texture3D StructureTexture { get; private set; }
    public Texture3D DensityTexture { get; private set; }

    public bool IsLoaded { get; private set; }

    public void LoadData()
    {
        string folder = Path.Combine(
            Application.streamingAssetsPath, "DESI_DR1"
        );

        string structurePath =
            Path.Combine(folder, structuresFile);

        string densityPath =
            Path.Combine(folder, densityFile);

        if (!File.Exists(structurePath))
            throw new FileNotFoundException(structurePath);

        if (!File.Exists(densityPath))
            throw new FileNotFoundException(densityPath);

        byte[] structureBytes =
            File.ReadAllBytes(structurePath);

        byte[] densityBytes =
            File.ReadAllBytes(densityPath);

        int voxelCount =
            resolution * resolution * resolution;

        if (structureBytes.Length != voxelCount)
            throw new Exception(
                $"Structure size incorrect: {structureBytes.Length}"
            );

        if (densityBytes.Length != voxelCount * 4)
            throw new Exception(
                $"Density size incorrect: {densityBytes.Length}"
            );

        if (!BitConverter.IsLittleEndian)
            throw new Exception(
                "Density data requires little-endian conversion."
            );

        if (!SystemInfo.SupportsTextureFormat(TextureFormat.R8))
            throw new Exception("R8 textures are unsupported.");

        if (!SystemInfo.SupportsTextureFormat(TextureFormat.RFloat))
            throw new Exception(
                "RFloat textures are unsupported on this device."
            );

        // ---------------------------------
        // STRUCTURE TEXTURE
        // ---------------------------------

        StructureTexture = new Texture3D(
            resolution,
            resolution,
            resolution,
            TextureFormat.R8,
            false
        );

        StructureTexture.name = "DESI_Structure_Volume";
        StructureTexture.filterMode = FilterMode.Point;
        StructureTexture.wrapMode = TextureWrapMode.Clamp;

        StructureTexture.SetPixelData<byte>(
            structureBytes, 0
        );

        StructureTexture.Apply(false, true);

        // ---------------------------------
        // DENSITY TEXTURE
        // ---------------------------------

        DensityTexture = new Texture3D(
            resolution,
            resolution,
            resolution,
            TextureFormat.RFloat,
            false
        );

        // Convert raw little-endian bytes to floats.
        float[] densityValues = new float[voxelCount];

        Buffer.BlockCopy(
            densityBytes,
            0,
            densityValues,
            0,
            densityBytes.Length
        );

        // Create density texture.
        DensityTexture = new Texture3D(
            resolution,
            resolution,
            resolution,
            TextureFormat.RFloat,
            false
        );

        DensityTexture.name = "DESI_Density_Volume";
        DensityTexture.filterMode = densityFilter;
        DensityTexture.wrapMode = TextureWrapMode.Clamp;

        DensityTexture.SetPixelData<float>(
            densityValues, 0
        );

        DensityTexture.Apply(false, true);

        IsLoaded = true;

        Debug.Log(
            $"DESI loaded: {voxelCount:N0} voxels per volume."
        );

        Debug.Log(
            $"Structure memory: {structureBytes.Length / 1048576f:F2} MB"
        );

        Debug.Log(
            $"Density memory: {densityBytes.Length / 1048576f:F2} MB"
        );
    }

    private void OnDestroy()
    {
        if (StructureTexture != null)
            Destroy(StructureTexture);

        if (DensityTexture != null)
            Destroy(DensityTexture);
    }
}

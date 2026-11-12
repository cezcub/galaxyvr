using UnityEngine;

public class DESIVisualizationController : MonoBehaviour
{
    public enum ViewMode
    {
        Galaxies = 0,
        Structures = 1,
        Combined = 2
    }

    [Header("Renderers")]
    public DESIGalaxyRenderer galaxyRenderer;
    public DESIVolumeRenderer volumeRenderer;

    [Header("Initial view")]
    public ViewMode initialMode = ViewMode.Combined;

    public ViewMode CurrentMode { get; private set; }

    private void Start()
    {
        SetMode(initialMode);
    }

    public void ShowGalaxies()
    {
        SetMode(ViewMode.Galaxies);
    }

    public void ShowStructures()
    {
        SetMode(ViewMode.Structures);
    }

    public void ShowCombined()
    {
        SetMode(ViewMode.Combined);
    }

    public void SetMode(ViewMode mode)
    {
        CurrentMode = mode;

        bool showGalaxies =
            mode == ViewMode.Galaxies ||
            mode == ViewMode.Combined;

        bool showStructures =
            mode == ViewMode.Structures ||
            mode == ViewMode.Combined;

        if (galaxyRenderer != null)
            galaxyRenderer.SetVisible(showGalaxies);

        if (volumeRenderer != null)
            volumeRenderer.SetVisible(showStructures);

        Debug.Log("DESI visualization mode: " + mode);
    }
}

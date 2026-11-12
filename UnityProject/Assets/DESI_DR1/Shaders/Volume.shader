Shader "DESI/CosmicWebVolume"
{
    Properties
    {
        _StructureTex ("Structure Volume", 3D) = "" {}
        _DensityTex ("Density Volume", 3D) = "" {}

        _VisualizationMode ("Visualization Mode", Integer) = 2
        _RaySteps ("Ray Steps", Integer) = 128

        _Opacity ("Opacity", Float) = 0.35
        _DensityMin ("Density Minimum", Float) = 0
        _DensityMax ("Density Maximum", Float) = 3
        _DensityThreshold ("Density Threshold", Float) = 0.02
        _Brightness ("Brightness", Float) = 1.5
    }

    SubShader
    {
        Tags
        {
            "RenderPipeline" = "UniversalPipeline"
            "Queue" = "Transparent"
            "RenderType" = "Transparent"
        }

        Pass
        {
            Name "DESIVolumeRayMarch"

            Cull Front
            ZWrite Off
            ZTest LEqual

            Blend One OneMinusSrcAlpha

            HLSLPROGRAM

            #pragma vertex Vert
            #pragma fragment Frag

            #pragma target 3.5

            #pragma multi_compile_instancing
            #pragma multi_compile _ STEREO_INSTANCING_ON STEREO_MULTIVIEW_ON

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            TEXTURE3D(_StructureTex);
            SAMPLER(sampler_StructureTex);

            TEXTURE3D(_DensityTex);
            SAMPLER(sampler_DensityTex);

            CBUFFER_START(UnityPerMaterial)

                int _VisualizationMode;
                int _RaySteps;

                float _Opacity;
                float _DensityMin;
                float _DensityMax;
                float _DensityThreshold;
                float _Brightness;

            CBUFFER_END

            struct Attributes
            {
                float4 positionOS : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 positionOS : TEXCOORD0;

                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings Vert(Attributes input)
            {
                Varyings output;

                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);

                output.positionOS = input.positionOS;

                output.positionCS =
                    TransformObjectToHClip(input.positionOS.xyz);

                return output;
            }

            // Ray-box intersection.
            bool IntersectBox(
                float3 origin,
                float3 direction,
                out float tNear,
                out float tFar
            )
            {
                float3 boxMin = float3(-0.5, -0.5, -0.5);
                float3 boxMax = float3( 0.5,  0.5,  0.5);

                float3 invDir = 1.0 / direction;

                float3 t0 = (boxMin - origin) * invDir;
                float3 t1 = (boxMax - origin) * invDir;

                float3 tSmall = min(t0, t1);
                float3 tLarge = max(t0, t1);

                tNear = max(
                    max(tSmall.x, tSmall.y),
                    tSmall.z
                );

                tFar = min(
                    min(tLarge.x, tLarge.y),
                    tLarge.z
                );

                return tFar >= max(tNear, 0.0);
            }

            // Classification color.
            float3 GetStructureColor(float id)
            {
                if (id < 0.5)
                    return float3(0.015, 0.025, 0.10);

                if (id < 1.5)
                    return float3(0.08, 0.30, 0.95);

                if (id < 2.5)
                    return float3(0.05, 0.85, 1.0);

                return float3(1.0, 0.55, 0.12);
            }

            // Structure opacity.
            float GetStructureOpacity(float id)
            {
                if (id < 0.5)
                    return 0.002;

                if (id < 1.5)
                    return 0.025;

                if (id < 2.5)
                    return 0.055;

                return 0.14;
            }

            half4 Frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

                // Camera position in local cube coordinates.
                float3 rayOrigin =
                    TransformWorldToObject(GetCameraPositionWS());

                // Ray toward the visible back face.
                float3 rayDirection = normalize(
                    input.positionOS - rayOrigin
                );

                float tNear;
                float tFar;

                if (!IntersectBox(
                    rayOrigin,
                    rayDirection,
                    tNear,
                    tFar
                ))
                {
                    discard;
                }

                float tStart = max(tNear, 0.0);

                if (tFar <= tStart)
                    discard;

                int steps = clamp(_RaySteps, 16, 256);

                float stepSize =
                    (tFar - tStart) / (float)steps;

                float4 accumulated = float4(0, 0, 0, 0);

                for (int i = 0; i < 256; i++)
                {
                    if (i >= steps)
                        break;

                    float t =
                        tStart + (i + 0.5) * stepSize;

                    float3 positionOS =
                        rayOrigin + rayDirection * t;

                    // Convert object-space position to texture UV.
                    float3 uv = positionOS + 0.5;

                    float structureID =
                        SAMPLE_TEXTURE3D(
                            _StructureTex,
                            sampler_StructureTex,
                            uv
                        ).r * 255.0;

                    float density =
                        SAMPLE_TEXTURE3D(
                            _DensityTex,
                            sampler_DensityTex,
                            uv
                        ).r;

                    // Normalize density using configurable range.
                    float densityNormalized = saturate(
                        (density - _DensityMin) /
                        max(_DensityMax - _DensityMin, 0.0001)
                    );

                    float3 structureColor =
                        GetStructureColor(structureID);

                    float structureAlpha =
                        GetStructureOpacity(structureID);

                    float3 color;
                    float alpha;

                    // MODE 0: CLASSIFIED STRUCTURES
                    if (_VisualizationMode == 0)
                    {
                        color = structureColor;
                        alpha = structureAlpha;
                    }

                    // MODE 1: CONTINUOUS DENSITY
                    else if (_VisualizationMode == 1)
                    {
                        color = lerp(
                            float3(0.02, 0.03, 0.25),
                            float3(0.0, 0.85, 1.0),
                            densityNormalized
                        );

                        color = lerp(
                            color,
                            float3(1.0, 0.48, 0.08),
                            smoothstep(
                                0.65,
                                1.0,
                                densityNormalized
                            )
                        );

                        alpha = smoothstep(
                            _DensityThreshold,
                            0.8,
                            densityNormalized
                        ) * 0.10;
                    }

                    // MODE 2: COMBINED
                    else
                    {
                        color = structureColor;

                        color *= lerp(
                            0.25,
                            2.0,
                            densityNormalized
                        );

                        alpha = structureAlpha *
                            lerp(
                                0.25,
                                2.0,
                                densityNormalized
                            );
                    }

                    color *= _Brightness;

                    alpha *= _Opacity;

                    // Scale opacity according to sampling interval.
                    // This reduces changes in appearance when
                    // the number of ray steps changes.
                    alpha = 1.0 - pow(
                        max(1.0 - saturate(alpha), 0.00001),
                        stepSize * 256.0
                    );

                    // Ignore low-density samples.
                    if (densityNormalized < _DensityThreshold)
                    {
                        if (_VisualizationMode != 0)
                            alpha = 0;
                    }

                    alpha = saturate(alpha);

                    // Front-to-back alpha compositing.
                    float contribution =
                        (1.0 - accumulated.a) * alpha;

                    accumulated.rgb +=
                        contribution * color;

                    accumulated.a += contribution;

                    // Early termination.
                    if (accumulated.a > 0.97)
                        break;
                }

                return half4(
                    accumulated.rgb,
                    accumulated.a
                );
            }

            ENDHLSL
        }
    }
}

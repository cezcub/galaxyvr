Shader "DESI/GalaxyPoints"
{
    Properties
    {
        _GalaxySize ("Galaxy Size", Float) = 0.025
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
            Name "GalaxyBillboards"

            Cull Off
            ZWrite Off
            ZTest LEqual
            Blend One OneMinusSrcAlpha

            HLSLPROGRAM

            #pragma vertex Vert
            #pragma fragment Frag

            #pragma target 4.5
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            StructuredBuffer<float4> _GalaxyPositions;

            float _GalaxySize;
            float _Brightness;

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;

                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings Vert(
                uint vertexID : SV_VertexID,
                uint instanceID : SV_InstanceID
            )
            {
                Varyings output;

                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);

                float4 galaxy =
                    _GalaxyPositions[instanceID];

                float3 centerWS =
                    TransformObjectToWorld(galaxy.xyz);

                float2 corners[6] =
                {
                    float2(-1, -1),
                    float2( 1, -1),
                    float2( 1,  1),

                    float2(-1, -1),
                    float2( 1,  1),
                    float2(-1,  1)
                };

                float2 corner = corners[vertexID];

                float3 right =
                    UNITY_MATRIX_I_V._m00_m10_m20;

                float3 up =
                    UNITY_MATRIX_I_V._m01_m11_m21;

                float3 positionWS =
                    centerWS +
                    right * corner.x * _GalaxySize +
                    up * corner.y * _GalaxySize;

                output.positionCS =
                    TransformWorldToHClip(positionWS);

                output.uv = corner;

                return output;
            }

            half4 Frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

                float radius = length(input.uv);

                float alpha =
                    1.0 - smoothstep(0.65, 1.0, radius);

                if (alpha < 0.01)
                    discard;

                float3 color = float3(
                    0.45,
                    0.75,
                    1.0
                ) * _Brightness;

                return half4(
                    color * alpha,
                    alpha
                );
            }

            ENDHLSL
        }
    }
}
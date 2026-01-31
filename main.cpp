#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>
#include <map>
#include <vector>

#include <nlohmann/json.hpp>

#include "io.h"
#include "dump.h"

using json = nlohmann::json;

// Wigner D-matrix for degree 1 (rotation around x-axis by -90° starting with negative index)
constexpr double d1_neg90[3][3] = {
    { 0.f,  -1.f,  0.f },
    { 1.f,  0.f,  0.f },
    { 0.f,  0.f,  1.f }
};

// Wigner D-matrix for degree 2 (rotation around x-axis by -90° starting with negative index)
constexpr double d2_neg90[5][5] = {
    { 0.f,  0.f,  0.f,  -1.f,  0.f },
    { 0.f, -1.f,  0.f,  0.f,  0.f },
    { 0.f,  0.f, -0.5f, 0.f, -0.866025403f },
    { 1.f,  0.f,  0.f,  0.f,  0.f },
    { 0.f,  0.f, -0.866025403f, 0.f,  0.5f }
};

// Wigner D-matrix for degree 3 (rotation around x-axis by -90° starting with negative index)
constexpr double d3_neg90[7][7] = {
    { 0.f, 0.f, 0.f, 0.79056942f, 0.f, -0.61237244f, 0.f},
    { 0.f, -1.f, 0.f, 0.f, 0.f, 0.f, 0.f },
    { 0.f,  0.f, 0.f, 0.61237244, 0.f, 0.79056942f, 0.f },
    { -0.79056942f, 0.f, -0.61237244, -0.f, 0.f, 0.f, -0.f},
    { 0.f, 0.f, 0.f, 0.f, -0.25f, 0.f, -0.96824584 },
    { 0.61237244f, 0.f, -0.79056942f, -0.f, 0.f, 0.f, 0.f},
    { 0.f, 0.f, 0.f, 0.f, -0.96824584f, 0.f, 0.25f  }
};

std::vector<float> rotateSH_XAxisPos90(const float* coefficients, std::uint32_t l)
{
    std::vector<float> result{};
    
    if (l >= 1u)
    {
        std::vector<double> in{coefficients[0u], coefficients[1u], coefficients[2u]};
        std::vector<double> out(3u, 0.0);

        for (std::uint32_t i = 0u; i < 3u; i++)
        {
            for (std::uint32_t j = 0u; j < 3u; j++)
            {
                out[i] += d1_neg90[i][j] * in[j];
            }
        }
        
        result.insert(result.end(), out.begin(), out.end());
    }
    
    if (l >= 2u)
    {
        std::vector<double> in{coefficients[3u + 0u], coefficients[3u + 1u], coefficients[3u + 2u], coefficients[3u + 3u], coefficients[3u + 4u]};
        std::vector<double> out(5u, 0.0);

        for (std::uint32_t i = 0u; i < 5u; i++)
        {
            for (std::uint32_t j = 0u; j < 5u; j++)
            {
                out[i] += d2_neg90[i][j] * in[j];
            }
        }
        
        result.insert(result.end(), out.begin(), out.end());
    }
    
    if (l >= 3u)
    {
        std::vector<double> in{coefficients[3u + 5u + 0u], coefficients[3u + 5u + 1u], coefficients[3u + 5u + 2u], coefficients[3u + 5u + 3u], coefficients[3u + 5u + 4u], coefficients[3u + 5u + 5u], coefficients[3u + 5u + 6u]};
        std::vector<double> out(7u, 0.0);

        for (std::uint32_t i = 0u; i < 7u; i++)
        {
            for (std::uint32_t j = 0u; j < 7u; j++)
            {
                out[i] += d3_neg90[i][j] * in[j];
            }
        }
        
        result.insert(result.end(), out.begin(), out.end());
    }
    
    return result;
}

std::vector<float> gather(const float* coefficients, std::uint32_t l)
{
    std::vector<float> result{};

    if (l >= 1u)
    {
        result.push_back(coefficients[0u]);
        result.push_back(coefficients[1u]);
        result.push_back(coefficients[2u]);
    }
    
    if (l >= 2u)
    {
        result.push_back(coefficients[3u + 0u]);
        result.push_back(coefficients[3u + 1u]);
        result.push_back(coefficients[3u + 2u]);
        result.push_back(coefficients[3u + 3u]);
        result.push_back(coefficients[3u + 4u]);
    }
    
    if (l >= 3u)
    {
        result.push_back(coefficients[3u + 5u + 0u]);
        result.push_back(coefficients[3u + 5u + 1u]);
        result.push_back(coefficients[3u + 5u + 2u]);
        result.push_back(coefficients[3u + 5u + 3u]);
        result.push_back(coefficients[3u + 5u + 4u]);
        result.push_back(coefficients[3u + 5u + 5u]);
        result.push_back(coefficients[3u + 5u + 6u]);
    }
    
    return result;
}

// Quaternion multiplication: result = q1 * q0, Indices: 0=x, 1=y, 2=z, 3=w
std::array<float, 4u> multiplyQuaternions(const std::array<float, 4u>& q1, const std::array<float, 4u>& q0)
{
    std::array<float, 4u> result;
    
    result[0] = q1[3]*q0[0] + q1[0]*q0[3] + q1[1]*q0[2] - q1[2]*q0[1]; // x
    result[1] = q1[3]*q0[1] - q1[0]*q0[2] + q1[1]*q0[3] + q1[2]*q0[0]; // y  
    result[2] = q1[3]*q0[2] + q1[0]*q0[1] - q1[1]*q0[0] + q1[2]*q0[3]; // z
    result[3] = q1[3]*q0[3] - q1[0]*q0[0] - q1[1]*q0[1] - q1[2]*q0[2]; // w
    
    return result;
}

int main(int argc, char* argv[])
{
    // Stores the related offset in the PLY file. Higher degrees are sorted by channel in PLY file, so this needs to be resolved differently for glTF.
    enum Attributes {
        POSITION,
        ROTATION,
        SCALE,
        OPACITY,
        SH_DEGREE_0_COEF_0,
        SH_DEGREE_HIGHER,
    };

    //

    if (argc < 2)
    {
        printf("Usage: ply2gltf filename [--convert] [--dump\n");

        return 0;
    }

    bool convert{false};
    bool dump{false};
    std::string loadname{argv[1]};
    for (int i = 2; i < argc; i++)
    {
        std::string flag{argv[i]};

        if (flag == "--convert")
        {
            convert = true;
        }
        else if (flag == "--dump")
        {
            dump = true;
        }
        else
        {
            printf("Usage: ply2gltf filename [--convert]\n");

            return 0;
        }
    }

    std::filesystem::path loadpath(loadname);
    auto stem = loadpath.stem().generic_string();
    auto extension = loadpath.extension().generic_string();

    std::string savenameJson{stem + ".gltf"};
    std::string savenameBinary{stem + ".bin"};

    std::string savenameDump{stem + "_dump.ply"};

    //
    // PLY loading
    //

    if (convert)
    {
        printf("Info: Converting from z-up right-handed to y-up right-handed coordinate system.\n");
    }
    else
    {
        printf("Info: No conversion and assuming y-up right-handed coordinate system.\n");
    }

    printf("Info: Loading '%s' ...\n", loadname.c_str());

    std::string ply = loadFile(loadname);
    if (ply.empty())
    {
        printf("Error: Could not load '%s'\n", loadname.c_str());

        return -1;
    }

    printf("Info: Loaded '%s'\n", loadname.c_str());

    auto index = ply.find("end_header\n");
    if (index == std::string::npos)
    {
        printf("Error: No header found\n");

        return -1;
    }

    // Extracted header required to setup the accessors from PLY.
    std::istringstream header{ply.substr(0, index + 11)};

    // Extracted PLY buffer for further processing the data.
    std::string binaryPly = ply.substr(index + 11);

    //
    // Setup glTF
    //

    printf("Info: Setting up glTF\n");

    // glTF binary

    std::string binary{};

    // glTF main object

    json glTF = json::object();

    //

    json asset = json::object();
    asset["version"] = "2.0";
    asset["generator"] = "3DGS PLY to glTF converter by Huawei";

    glTF["asset"] = asset;

    //

    json extensionsUsed = json::array();
    extensionsUsed.push_back("KHR_gaussian_splatting");

    glTF["extensionsUsed"] = extensionsUsed;

    json extensionsRequired = json::array();
    extensionsRequired.push_back("KHR_gaussian_splatting");

    glTF["extensionsRequired"] = extensionsRequired;

    //

    json buffers = json::array();

    json buffer = json::object();
    buffer["uri"] = savenameBinary;
    // byteLength will be later set.

    buffers.push_back(buffer);

    glTF["buffers"] = buffers;

    //

    json bufferViews = json::array();

    json bufferView = json::object();
    bufferView["buffer"] = 0;
    // byteLength will be later set.
    // byteStride will be later set.
    bufferView["target"] = 34962;

    bufferViews.push_back(bufferView);

    glTF["bufferViews"] = bufferViews;

    //

    json accessors = json::array();

    std::uint32_t byteOffset{0u};
    for (std::uint32_t i = 0u; i < 5u; i++)
    {
        json accessor = json::object();
        accessor["bufferView"] = 0;
        accessor["byteOffset"] = byteOffset;
        accessor["componentType"] = 5126;
        // count will be later set.

        if (static_cast<Attributes>(i) == POSITION)
        {
            accessor["name"] = "POSITION";
            accessor["type"] = "VEC3";

            byteOffset += 3u * sizeof(float);
        }
        else if (static_cast<Attributes>(i) == ROTATION)
        {
            accessor["name"] = "ROTATION";
            accessor["type"] = "VEC4";

            byteOffset += 4u * sizeof(float);
        }
        else if (static_cast<Attributes>(i) == SCALE)
        {
            accessor["name"] = "SCALE";
            accessor["type"] = "VEC3";

            byteOffset += 3u * sizeof(float);
        }
        else if (static_cast<Attributes>(i) == OPACITY)
        {
            accessor["name"] = "OPACITY";
            accessor["type"] = "SCALAR";

            byteOffset += 1u * sizeof(float);
        }
        else if (static_cast<Attributes>(i) == SH_DEGREE_0_COEF_0)
        {
            accessor["name"] = "SH_DEGREE_0_COEF_0";
            accessor["type"] = "VEC3";

            byteOffset += 3u * sizeof(float);
        }
 
        accessors.push_back(accessor);
    }

    // other accessors will be later set.

    glTF["accessors"] = accessors;

    //

    json meshes = json::array();

    json mesh = json::object();
    mesh["primitives"] = json::array();

    json primitive = json::object();
    primitive["mode"] = 0;
    primitive["attributes"] = json::object();
    primitive["attributes"]["POSITION"] = 0u;
    primitive["attributes"]["KHR_gaussian_splatting:ROTATION"] = 1u;
    primitive["attributes"]["KHR_gaussian_splatting:SCALE"] = 2u;
    primitive["attributes"]["KHR_gaussian_splatting:OPACITY"] = 3u;
    primitive["attributes"]["KHR_gaussian_splatting:SH_DEGREE_0_COEF_0"] = 4u;
    // other attributes will be later set.
    primitive["extensions"] = json::object();
    primitive["extensions"]["KHR_gaussian_splatting"] = json::object();
    primitive["extensions"]["KHR_gaussian_splatting"]["kernel"] = "ellipse";
    primitive["extensions"]["KHR_gaussian_splatting"]["colorSpace"] = "srgb_rec709_display";

    mesh["primitives"].push_back(primitive);

    meshes.push_back(mesh);

    glTF["meshes"] = meshes;

    //

    json nodes = json::array();

    json node = json::object();
    node["mesh"] = 0;

    nodes.push_back(node);

    glTF["nodes"] = nodes;

    //

    json scenes = json::array();

    json scene = json::object();
    scene["nodes"] = json::array();
    scene["nodes"].push_back(0);

    scenes.push_back(scene);

    glTF["scenes"] = scenes;

    glTF["scene"] = 0;

    std::uint32_t count{0u};
    std::uint32_t byteStride{0u};

    //
    // Processing PLY header and binary data.
    //

    printf("Info: Parsing PLY header\n");

    bool isPly{false};
    bool isBinaryLittleEndian{false};

    std::uint32_t sourceByteStride{0u};
    std::map<Attributes, std::uint32_t> sourceByteOffsets{};

    std::uint32_t rests{0u};

    std::string line{};
    while (std::getline(header, line))
    {
        if (line == "ply")
        {
            isPly = true;
        }
        else if (line.find("format binary_little_endian") != std::string::npos)
        {
            isBinaryLittleEndian = true;
        }
        else if (line.find("element vertex") != std::string::npos)
        {
#ifdef _WIN32
            auto result = sscanf_s(line.c_str(), "element vertex %u", &count);
#else
            auto result = std::sscanf(line.c_str(), "element vertex %u", &count);
#endif
            if (result < 0)
            {
                return -1;
            }
        }
        else if (line == "end_header")
        {
            break;
        }
        else
        {
            char componentType[256u];
            char name[256u];

#ifdef _WIN32
            auto result = sscanf_s(line.c_str(), "property %s %s", componentType, 256u, name, 256u);
#else
            auto result = std::sscanf(line.c_str(), "property %255s %255s", componentType, name);
#endif
            if (result < 0)
            {
                return -1;
            }

            std::string checkComponentType{componentType};
            if (checkComponentType != "float")
            {
                printf("Error: Unknown component type '%s'\n", checkComponentType.c_str());

                return -1;
            }

            std::string checkName{name};

            if (checkName == "nx")
            {
                // Not storing, however source byte stride needs to be adapted.
                sourceByteStride += 3u * sizeof(float);

                continue;
            }
            else if (checkName == "x")
            {
                sourceByteOffsets[Attributes::POSITION] = sourceByteStride;

                sourceByteStride += 3u * sizeof(float);
            }
            else if (checkName == "rot_0")
            {
                sourceByteOffsets[Attributes::ROTATION] = sourceByteStride;

                sourceByteStride += 4u * sizeof(float);
            }
            else if (checkName == "scale_0")
            {
                sourceByteOffsets[Attributes::SCALE] = sourceByteStride;

                sourceByteStride += 3u * sizeof(float);
            }
            else if (checkName == "opacity")
            {
                sourceByteOffsets[Attributes::OPACITY] = sourceByteStride;
            
                sourceByteStride += 1u * sizeof(float);
            }
            else if (checkName == "f_dc_0")
            {
                sourceByteOffsets[Attributes::SH_DEGREE_0_COEF_0] = sourceByteStride;

                sourceByteStride += 3u * sizeof(float);
            }
            else if (checkName == "f_rest_0")
            {
                sourceByteOffsets[SH_DEGREE_HIGHER] = sourceByteStride;

                sourceByteStride += 1u * sizeof(float);

                rests++;
            }
            else if (checkName.starts_with("f_rest_"))
            {
                // Higher degrees are differently stored in PLY, so the general offset it sufficient.

                sourceByteStride += 1u * sizeof(float);

                rests++;
            }
            else
            {
                // Note: Assuming, that PLY file is correctly packed e.g. x then y then z and sorted e.g. 0 then 1 and so on. Swizzling the rotation does not affect this and happens later.
                continue;
            }
        }
    }

    if (!isPly || !isBinaryLittleEndian || !count || !sourceByteOffsets.contains(Attributes::POSITION) || !sourceByteOffsets.contains(Attributes::SCALE) || !sourceByteOffsets.contains(Attributes::OPACITY) || !sourceByteOffsets.contains(Attributes::ROTATION) || !sourceByteOffsets.contains(Attributes::SH_DEGREE_0_COEF_0))
    {
        printf("Error: Can not process `%s` file\n", loadname.c_str());

        return -1;
    }

    // Update count on all current accessors.

    glTF["accessors"][0u]["count"] = count;
    glTF["accessors"][1u]["count"] = count;
    glTF["accessors"][2u]["count"] = count;
    glTF["accessors"][3u]["count"] = count;
    glTF["accessors"][4u]["count"] = count;

    // Depending on rests entries in the PLY file, deduct the degree.
    std::uint32_t l{0u};
    if (rests == 0u)
    {
        // Nothing for now
    }
    else if (rests == 3u * 3u)
    {
        l = 1u;
    }
    else if (rests == 3u * 3u + 5u * 3u)
    {
        l = 2u;
    }
    else if (rests == 3u * 3u + 5u * 3u + 7u * 3u)
    {
        l = 3u;
    }
    else
    {
        printf("Error: Unsupported amount of rest entries\n");

        return -1;
    }

    // Generate accessors depending on degrees.
    for (std::uint32_t current_l = 1u; current_l <= l; current_l++)
    {
        for (std::uint32_t current_n = 0u; current_n < 1u + 2u * current_l; current_n++)
        {
            std::string current_name{"SH_DEGREE_" + std::to_string(current_l) + "_COEF_" + std::to_string(current_n)};

            json accessor = json::object();
            accessor["name"] = current_name;
            accessor["bufferView"] = 0;
            accessor["byteOffset"] = byteOffset;
            accessor["componentType"] = 5126;
            accessor["count"] = count;
            accessor["type"] = "VEC3";

            glTF["meshes"][0u]["primitives"][0u]["attributes"]["KHR_gaussian_splatting:" + current_name] = glTF["accessors"].size();

            glTF["accessors"].push_back(accessor);

            byteOffset += 3u * sizeof(float);
        }
    }

    byteStride = byteOffset;

    // Final buffer size can be calculated.
    binary.resize(byteStride * count);

    printf("Info: Processing PLY binary data\n");

    // Loop through vertices and by our given order how we store the attributes.
    for (std::uint32_t vertex = 0u; vertex < count; vertex++)
    {
        byteOffset = 0u;

        {
            // POSITION
            const float* sourceData = reinterpret_cast<const float*>(binaryPly.data() + sourceByteStride * vertex + sourceByteOffsets[POSITION]);
            float* data = reinterpret_cast<float*>(binary.data() + byteStride * vertex + byteOffset);

            float x{sourceData[0u]}; 
            float y{sourceData[1u]}; 
            float z{sourceData[2u]}; 

            if (convert)
            {
                // Convert from right-handed z-up to right-handed y-up coordinate system. -90 degree rotation results in this swizzle.
                data[0u] = x;
                data[1u] = z;
                data[2u] = -y;
            }
            else
            {
                data[0u] = x;
                data[1u] = y;
                data[2u] = z;
            }

            byteOffset += 3u * sizeof(float);
        }

        {
            // ROTATION
            const float* sourceData = reinterpret_cast<const float*>(binaryPly.data() + sourceByteStride * vertex + sourceByteOffsets[ROTATION]);
            float* data = reinterpret_cast<float*>(binary.data() + byteStride * vertex + byteOffset);

            // Need to swizzle the quaternion data because of layout.
            float w{sourceData[0u]}; 
            float x{sourceData[1u]}; 
            float y{sourceData[2u]}; 
            float z{sourceData[3u]}; 

            // Also normalize it, as not given by PLY.
            float norm = std::sqrt(x*x + y*y + z*z + w*w);
            if (norm == 0.0f)
            {
                printf("Error: Invalid quaternion\n");

                return -1;
            }

            x = x / norm;
            y = y / norm;
            z = z / norm;
            w = w / norm;

            if (convert)
            {
                // Rotate -90 degree around x-axis, to convert from right-handed z-up to right-handed y-up coordinate system.
                auto rotated = multiplyQuaternions({-0.7071, 0.0, 0.0, 0.7071}, {x, y, z, w});

                data[0u] = rotated[0];
                data[1u] = rotated[1];
                data[2u] = rotated[2];
                data[3u] = rotated[3];
            }
            else
            {
                data[0u] = x;
                data[1u] = y;
                data[2u] = z;
                data[3u] = w;
            }

            byteOffset += 4u * sizeof(float);
        }

        {
            // SCALE
            const float* sourceData = reinterpret_cast<const float*>(binaryPly.data() + sourceByteStride * vertex + sourceByteOffsets[SCALE]);
            float* data = reinterpret_cast<float*>(binary.data() + byteStride * vertex + byteOffset);

            float x{sourceData[0u]}; 
            float y{sourceData[1u]}; 
            float z{sourceData[2u]}; 

            // No rotation required, as scale impacted by rotation.
            data[0u] = x;
            data[1u] = y;
            data[2u] = z;

            byteOffset += 3u * sizeof(float);
        }

        {
            // OPACITY
            const float* sourceData = reinterpret_cast<const float*>(binaryPly.data() + sourceByteStride * vertex + sourceByteOffsets[OPACITY]);
            float* data = reinterpret_cast<float*>(binary.data() + byteStride * vertex + byteOffset);

            // Sigmoid function needs to be applied before storing.
            const float opacity = *sourceData;
            *data = 1.0f / (1.0f + std::exp(-opacity));

            byteOffset += 1u * sizeof(float);
        }

        {
            // SH_DEGREE_0_COEF_0
            const float* sourceData = reinterpret_cast<const float*>(binaryPly.data() + sourceByteStride * vertex + sourceByteOffsets[SH_DEGREE_0_COEF_0]);
            float* data = reinterpret_cast<float*>(binary.data() + byteStride * vertex + byteOffset);

            // No rotation required, as identity.
            data[0u] = sourceData[0u];
            data[1u] = sourceData[1u];
            data[2u] = sourceData[2u];

            byteOffset += 3u * sizeof(float);
        }

        // Resolve for higher degrees.
        {
            // Offset at beginning to all bands.
            const float* sourceData = reinterpret_cast<const float*>(binaryPly.data() + sourceByteStride * vertex + sourceByteOffsets[SH_DEGREE_HIGHER]);

            // Offset between coefficient sets depending on degree.
            std::uint32_t sh_offset{0u};
            if (l == 1u)
            {
                sh_offset = 3u;
            }
            else if (l == 2u)
            {
                sh_offset = 3u + 5u;
            }
            else if (l == 3u)
            {
                sh_offset = 3u + 5u + 7u;
            }

            std::vector<float> r = gather(&sourceData[0u * sh_offset], l);
            std::vector<float> g = gather(&sourceData[1u * sh_offset], l);
            std::vector<float> b = gather(&sourceData[2u * sh_offset], l);

            if (convert)
            {
                // Rotate the spherical harmonics as well by -90 degrees around x-axis with optimized Wigner d-Matrix.
                r = rotateSH_XAxisPos90(r.data(), l);
                g = rotateSH_XAxisPos90(g.data(), l);
                b = rotateSH_XAxisPos90(b.data(), l);
            }

            std::uint32_t band_offset{0u};
            for (std::uint32_t current_l = 1u; current_l <= l; current_l++)
            {
                for (std::uint32_t current_n = 0u; current_n < 1u + 2u * current_l; current_n++)
                {
                    float* data = reinterpret_cast<float*>(binary.data() + byteStride * vertex + byteOffset);

                    data[0u] = r[band_offset + current_n];
                    data[1u] = g[band_offset + current_n];
                    data[2u] = b[band_offset + current_n];

                    byteOffset += 3u * sizeof(float);
                }

                band_offset += 1u + 2u * current_l; 
            }
        }
    }

    // End of PLY specific code.

    //
    // Finalizing glTF setup.
    //

    // byteLength can now be set
    glTF["buffers"][0]["byteLength"] = binary.size();

    // byteLength and byteStride can now be set
    glTF["bufferViews"][0u]["byteLength"] = binary.size();
    glTF["bufferViews"][0u]["byteStride"] = byteStride;

    // Gather min and max for POSITION, as required by specification.
    float min_position[3]{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    float max_position[3]{std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min()};
    for (std::uint32_t vertex = 0u; vertex < count; vertex++)
    {
        // Position is written at first position, so no offset required.
        const float* data = reinterpret_cast<const float*>(binary.data() + byteStride * vertex);

        for (std::uint32_t i = 0u; i < 3u; i++)
        {
            if (data[i] < min_position[i])
            {
                min_position[i] = data[i];
            }
            if (data[i] > max_position[i])
            {
                max_position[i] = data[i];
            }
        }
    }

    // Position is also first accessor.
    glTF["accessors"][0u]["min"] = json::array();
    glTF["accessors"][0u]["max"] = json::array();
    for (std::uint32_t i = 0u; i < 3u; i++)
    {
        glTF["accessors"][0u]["min"].push_back(min_position[i]);
        glTF["accessors"][0u]["max"].push_back(max_position[i]);
    }

    //
    // Storing to disk.
    //

    if (!saveFile(binary, savenameBinary))
    {
        printf("Error: Could not save '%s'\n", savenameBinary.c_str());

        return -1;
    }

    printf("Info: Saved '%s'\n", savenameBinary.c_str());

    if (!saveFile(glTF.dump(3), savenameJson))
    {
        printf("Error: Could not save '%s'\n", savenameJson.c_str());

        return -1;
    }

    printf("Info: Saved '%s'\n", savenameJson.c_str());

    printf("Info: Success\n");

    if (dump)
    {
        std::string plyDump = dumpPly(binary, count, byteStride, l);

        if (plyDump.empty())
        {
            printf("Error: Could not create PLY dump\n");

            return -1;
        }            

        if (!saveFile(plyDump, savenameDump))
        {
            printf("Error: Could not save '%s'\n", savenameDump.c_str());

            return -1;
        }

        printf("Info: Saved '%s'\n", savenameDump.c_str());
    }

	return 0;
}

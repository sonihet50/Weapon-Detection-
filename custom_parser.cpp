#include <vector>
#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseYolo(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    if (outputLayersInfo.empty()) return false;
    
    const float* data = (const float*)outputLayersInfo[0].buffer;
    int num_boxes = outputLayersInfo[0].inferDims.d[0]; 
    
    for (int i = 0; i < num_boxes; i++) {
        // Ultralytics TopK ONNX outputs: [x1, y1, x2, y2, confidence, class_id]
        float x1 = data[i * 6 + 0];
        float y1 = data[i * 6 + 1];
        float x2 = data[i * 6 + 2];
        float y2 = data[i * 6 + 3];
        float conf = data[i * 6 + 4];
        int class_id = (int)data[i * 6 + 5];
        
        if (conf >= detectionParams.perClassPreclusterThreshold[0]) {
            NvDsInferParseObjectInfo obj;
            obj.classId = class_id;
            obj.detectionConfidence = conf;
            
            // DeepStream requires Top-Left and dimensions
            obj.left = x1;
            obj.top = y1;
            obj.width = x2 - x1;
            obj.height = y2 - y1;
            
            objectList.push_back(obj);
        }
    }
    return true;
}

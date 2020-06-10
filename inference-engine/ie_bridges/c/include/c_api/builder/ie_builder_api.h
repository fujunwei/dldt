#ifndef IE_BUILDER_API_H
#define IE_BUILDER_API_H

#include <c_api/ie_c_api.h>

typedef struct ie_model ie_model_t;
typedef struct ie_compilation ie_compilation_t;
typedef struct ie_execution ie_execution_t;

typedef struct ie_operand {
    int32_t type;
    int32_t dimensionCount;
    const uint32_t* dimensions;
    float scale;
    int32_t zeroPoint;
} ie_operand_t;

/**
 * @brief Constructs Inference Engine Model instance. Use the ie_model_free() method to free memory.
 * @ingroup Model
 * @param model A pointer to the newly created ie_model_t.
 * @return Status code of the operation: OK(0) for success.
 */
INFERENCE_ENGINE_C_API(int32_t) ie_model_create(ie_model_t **model);

/**
 * @brief Releases memory occupied by model.
 * @ingroup Model
 * @param model A pointer to the model to free memory.
 */
INFERENCE_ENGINE_C_API(void) ie_model_free(ie_model_t *model);

/**
 * @brief Add operand to model.
 * @ingroup Model
 * @param model A pointer to the specified ie_model_t.
 * @param operand A pointer to ie_operand_t that will be add to model.
 * @return Status code of the operation: OK(0) for success.
 */
INFERENCE_ENGINE_C_API(IE_NODISCARD int32_t) ie_model_add_operand(ie_model_t *model,
                                                                    ie_operand_t* operand);

/**
 * @brief Set operand value.
 * @ingroup Model
 * @param model A pointer to the specified ie_model_t.
 * @param index the index of operand.
 * @param buffer the buffer of operand.
 * @param length the length of operand.
 * @return Status code of the operation: OK(0) for success.
 */
INFERENCE_ENGINE_C_API(IE_NODISCARD int32_t) ie_model_set_operand_value(ie_model_t *model,
                                                                            uint32_t index,
                                                                            const void* buffer,
                                                                            size_t length);

/**
 * @brief Add operation to model.
 * @ingroup Model
 * @param model A pointer to the specified ie_model_t.
 * @param inputCount the count of input.
 * @param inputs the operand of index for inputs.
 * @param outputCount the count of output.
 * @param outputs the operand of index for outputs.
 * @return Status code of the operation: OK(0) for success.
 */
INFERENCE_ENGINE_C_API(IE_NODISCARD int32_t) ie_model_add_operation(ie_model_t *model,
                                                                        int32_t type,
                                                                        uint32_t inputCount,
                                                                        const uint32_t* inputs,
                                                                        uint32_t outputCount,
                                                                        const uint32_t* outputs);

/**
 * @brief Add operation to model.
 * @ingroup Model
 * @param model A pointer to the specified ie_model_t.
 * @param inputCount the count of input.
 * @param inputs the operand of index for inputs.
 * @param outputCount the count of output.
 * @param outputs the operand of index for outputs.
 * @return Status code of the operation: OK(0) for success.
 */
INFERENCE_ENGINE_C_API(IE_NODISCARD int32_t) ie_model_identify_inputs_outputs(ie_model_t *model,
                                                                        uint32_t inputCount,
                                                                        const uint32_t* inputs,
                                                                        uint32_t outputCount,
                                                                        const uint32_t* outputs);

/**
 * @brief Create Compilation for the model.
 * @ingroup Compilation
 * @param model A pointer to the newly created ie_compilation_t.
 * @return Status code of the operation: OK(0) for success.
 */
INFERENCE_ENGINE_C_API(IE_NODISCARD int32_t) ie_compilation_create(ie_model_t* model, ie_compilation_t**compliation);


/**
 * @brief Set prefence to the Compilation.
 * @ingroup Compilation
 * @param compliation A pointer to the specified ie_compilation_t.
 * @param preference preference.
 * @return Status code of the operation: OK(0) for success.
 */
INFERENCE_ENGINE_C_API(IE_NODISCARD int32_t) ie_compilation_set_preference(
        ie_compilation_t* compliation, int32_t preference);

/**
 * @brief Start to compile the model.
 * @ingroup Compilation
 * @param compliation A pointer to the specified ie_compilation_t.
 * @return Status code of the operation: OK(0) for success.
 */
INFERENCE_ENGINE_C_API(IE_NODISCARD int32_t) ie_compilation_finish(
        ie_compilation_t* compliation);

/**
 * @brief Releases memory occupied by compilation.
 * @ingroup Model
 * @param model A pointer to the compilation to free memory.
 */
INFERENCE_ENGINE_C_API(void) ie_compilation_free(ie_compilation_t *compilation);

/**
 * @brief Create execution for the model.
 * @ingroup Execution
 * @param compliation A pointer to the specified ie_compilation_t.
 * @return Status code of the operation: OK(0) for success.
 */
INFERENCE_ENGINE_C_API(IE_NODISCARD int32_t) ie_execution_create(
        ie_compilation_t* compliation, ie_execution_t** execution);

/**
 * @brief Set input data for the model.
 * @ingroup Execution
 * @param compliation A pointer to the specified ie_execution_t.
 * @return Status code of the operation: OK(0) for success.
 */
INFERENCE_ENGINE_C_API(IE_NODISCARD int32_t) ie_execution_set_input(
        ie_execution_t* execution, uint32_t index, void* buffer, uint32_t length);

/**
 * @brief Set output data for the model.
 * @ingroup Execution
 * @param compliation A pointer to the specified ie_execution_t.
 * @return Status code of the operation: OK(0) for success.
 */
INFERENCE_ENGINE_C_API(IE_NODISCARD int32_t) ie_execution_set_output(
        ie_execution_t* execution, uint32_t index, void* buffer, uint32_t length);

/**
 * @brief Start compute the model.
 * @ingroup Execution
 * @param compliation A pointer to the specified ie_execution_t.
 * @return Status code of the operation: OK(0) for success.
 */
INFERENCE_ENGINE_C_API(IE_NODISCARD int32_t) ie_execution_start_compute(
        ie_execution_t* execution);

/**
 * @brief Releases memory occupied by execution.
 * @ingroup Model
 * @param model A pointer to the execution to free memory.
 */
INFERENCE_ENGINE_C_API(void) ie_execution_free(ie_execution_t *execution);
#endif // IE_BUILDER_API_H
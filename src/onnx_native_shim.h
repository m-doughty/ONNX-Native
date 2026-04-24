/*
 * onnx_native_shim.h — public surface of libonnx_native_shim.
 *
 * This header exists for documentation only. Raku's NativeCall
 * doesn't read C headers; lib/ONNX/Native/FFI.rakumod declares
 * each symbol below directly. Keep the signatures here in sync
 * with that file.
 *
 * Error convention:
 *   - Every function returns an int (OrtErrorCode, 0 == ORT_OK).
 *   - On error, *out_error is set to a malloc'd UTF-8 string that
 *     the caller must free with onnx_shim_free_error. On success,
 *     *out_error is left set to NULL.
 *   - Callers MAY pass NULL for out_error to discard error text;
 *     the return code is still authoritative.
 *
 * Tensor data ownership:
 *   - onnx_shim_create_tensor does NOT copy the data buffer. The
 *     caller must keep the backing memory alive for the lifetime
 *     of the returned OrtValue. Release the OrtValue with
 *     onnx_shim_release_value; the data buffer remains the
 *     caller's to free.
 *   - onnx_shim_tensor_data returns a pointer into the OrtValue's
 *     own memory (owned or borrowed depending on how the value
 *     was constructed). The pointer is valid until the OrtValue
 *     is released.
 */

#ifndef ONNX_NATIVE_SHIM_H
#define ONNX_NATIVE_SHIM_H

#include <stddef.h>
#include <stdint.h>

/* MSVC doesn't export symbols from DLLs by default; every public
 * function needs __declspec(dllexport) at its definition. GCC /
 * Clang export all non-static symbols by default for shared libs,
 * so this macro is a no-op there. Both the declaration header and
 * the definitions in the .c file use ONNX_SHIM_EXPORT. */
#if defined(_WIN32) || defined(__CYGWIN__)
#  define ONNX_SHIM_EXPORT __declspec(dllexport)
#else
#  define ONNX_SHIM_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque types — the shim forwards these through to ORT's own
 * OrtEnv / OrtSession / OrtSessionOptions / OrtValue. Callers
 * (Raku) treat them as opaque Pointer[]s. */
typedef struct OrtEnv            OrtEnv;
typedef struct OrtSession        OrtSession;
typedef struct OrtSessionOptions OrtSessionOptions;
typedef struct OrtValue          OrtValue;

/* --- Lifecycle --- */

ONNX_SHIM_EXPORT int onnx_shim_init(const char *log_id,
                                    OrtEnv **out_env,
                                    char **out_error);
ONNX_SHIM_EXPORT void onnx_shim_release_env(OrtEnv *env);

ONNX_SHIM_EXPORT int onnx_shim_create_session_options(OrtSessionOptions **out,
                                                      char **out_error);
ONNX_SHIM_EXPORT void onnx_shim_release_session_options(OrtSessionOptions *options);

/* provider_name ∈ {"cpu", "coreml", "cuda", "dml"}. "cpu" is a
 * no-op (the CPU provider is always registered). Unknown or
 * compiled-out providers return ORT_NOT_IMPLEMENTED with a
 * descriptive out_error. The flags argument is passed through
 * unchanged to the underlying provider factory (e.g. CoreML
 * coreml_flags, CUDA device_id). */
ONNX_SHIM_EXPORT int onnx_shim_enable_provider(OrtSessionOptions *options,
                                               const char *provider_name,
                                               int64_t flags,
                                               char **out_error);

ONNX_SHIM_EXPORT int onnx_shim_create_session_from_path(OrtEnv *env,
                                                        const char *model_path,
                                                        OrtSessionOptions *options,
                                                        OrtSession **out,
                                                        char **out_error);
ONNX_SHIM_EXPORT int onnx_shim_create_session_from_buffer(OrtEnv *env,
                                                          const void *buf,
                                                          size_t len,
                                                          OrtSessionOptions *options,
                                                          OrtSession **out,
                                                          char **out_error);
ONNX_SHIM_EXPORT void onnx_shim_release_session(OrtSession *session);

/* --- Session introspection --- */

ONNX_SHIM_EXPORT int onnx_shim_session_input_count(OrtSession *session,
                                                   size_t *out,
                                                   char **out_error);
ONNX_SHIM_EXPORT int onnx_shim_session_output_count(OrtSession *session,
                                                    size_t *out,
                                                    char **out_error);
ONNX_SHIM_EXPORT int onnx_shim_session_input_name(OrtSession *session,
                                                  size_t idx,
                                                  char **out_name,
                                                  char **out_error);
ONNX_SHIM_EXPORT int onnx_shim_session_output_name(OrtSession *session,
                                                   size_t idx,
                                                   char **out_name,
                                                   char **out_error);
/* Free a name returned by input_name/output_name via ORT's
 * default allocator (the same allocator they were allocated
 * from internally). Always safe to call with NULL. */
ONNX_SHIM_EXPORT void onnx_shim_free_name(char *name);

/* Type info: returns element type + dimension rank + up to
 * shape_cap dimension values. out_rank is always populated on
 * success; if shape_cap < out_rank, the first shape_cap dims
 * are written and the caller should re-allocate and call again.
 * Dimension values of -1 denote symbolic / dynamic dimensions. */
ONNX_SHIM_EXPORT int onnx_shim_session_input_type_info(OrtSession *session,
                                                       size_t idx,
                                                       int32_t *out_elem_type,
                                                       size_t *out_rank,
                                                       int64_t *out_shape,
                                                       size_t shape_cap,
                                                       char **out_error);
ONNX_SHIM_EXPORT int onnx_shim_session_output_type_info(OrtSession *session,
                                                        size_t idx,
                                                        int32_t *out_elem_type,
                                                        size_t *out_rank,
                                                        int64_t *out_shape,
                                                        size_t shape_cap,
                                                        char **out_error);

/* --- Tensors --- */

/* Create a tensor that borrows `data` for its lifetime.
 * elem_type is an ONNXTensorElementDataType value (1=FLOAT, 6=INT32,
 * 7=INT64 — see the header for the full enum). */
ONNX_SHIM_EXPORT int onnx_shim_create_tensor(const void *data,
                                             size_t byte_len,
                                             const int64_t *shape,
                                             size_t rank,
                                             int32_t elem_type,
                                             OrtValue **out,
                                             char **out_error);
ONNX_SHIM_EXPORT void onnx_shim_release_value(OrtValue *value);

/* Read shape from an OrtValue. Same semantics as the type_info
 * calls above — out_rank always set, shape populated up to cap. */
ONNX_SHIM_EXPORT int onnx_shim_tensor_shape(OrtValue *value,
                                            int32_t *out_elem_type,
                                            size_t *out_rank,
                                            int64_t *out_shape,
                                            size_t shape_cap,
                                            char **out_error);
/* Writes a pointer into the OrtValue's data to *out_data, and the
 * data's byte length to *out_byte_len. The pointer is valid until
 * the OrtValue is released; the caller does NOT free it. */
ONNX_SHIM_EXPORT int onnx_shim_tensor_data(OrtValue *value,
                                           void **out_data,
                                           size_t *out_byte_len,
                                           char **out_error);

/* --- Run --- */

/* Run inference. inputs / outputs are parallel arrays of length
 * num_inputs / num_outputs. On success, outputs[i] is set to a
 * fresh OrtValue* that the caller must release. On failure, no
 * output values are written. */
ONNX_SHIM_EXPORT int onnx_shim_run(OrtSession *session,
                                   const char *const *input_names,
                                   OrtValue *const *inputs,
                                   size_t num_inputs,
                                   const char *const *output_names,
                                   OrtValue **outputs,
                                   size_t num_outputs,
                                   char **out_error);

/* --- Error helper --- */

/* Free an error string populated by any of the functions above. */
ONNX_SHIM_EXPORT void onnx_shim_free_error(char *err);

/* --- Build-time info --- */

/* Returns the ORT_API_VERSION the shim was compiled against. Used
 * by the Raku side to sanity-check ABI compatibility. */
ONNX_SHIM_EXPORT int onnx_shim_api_version(void);

#ifdef __cplusplus
}
#endif

#endif /* ONNX_NATIVE_SHIM_H */

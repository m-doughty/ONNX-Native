/*
 * onnx_native_shim.c — flat C wrappers around ONNX Runtime's OrtApi
 * struct, for consumption by Raku NativeCall.
 *
 * ORT's public C API is exposed through a large struct of function
 * pointers obtained via OrtGetApiBase()->GetApi(ORT_API_VERSION).
 * Binding that through NativeCall directly would mean modelling the
 * struct as a ~300-field CStruct of Pointers and nativecasting each
 * field's signature per call site — noisy and error-prone. This
 * shim instead fetches the OrtApi* once, caches it in a static, and
 * exposes flat extern-C functions that Raku binds with plain
 * `is native('onnx_native_shim')`.
 *
 * The shim also normalises ORT's error convention: every ORT call
 * returns an OrtStatus* (NULL on success, allocated on failure);
 * the shim captures errors once here and returns (int code,
 * char* message) to the caller. The Raku side only has to know
 * one error convention, not two.
 *
 * Execution-provider registration lives here because it's the
 * least portable part of ORT: the CoreML / CUDA / DirectML
 * append-provider functions are free symbols (not on OrtApi), and
 * which ones are actually linked depends on the prebuilt variant.
 * The shim hides that per-platform noise behind a uniform
 * onnx_shim_enable_provider("coreml"|"cuda"|...) entry point.
 */

#include "onnx_native_shim.h"
#include "onnxruntime_c_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include "coreml_provider_factory.h"
#endif

#ifdef _WIN32
#include <windows.h>
#endif

/* --- ORT API accessor --- */

/* Cached pointer to the OrtApi function table. Populated once on
 * the first shim call that needs it (thread-safe because pointer
 * writes on the platforms we support are atomic at word size, and
 * the value is deterministic — any concurrent-init races write the
 * same pointer). */
static const OrtApi *g_ort = NULL;

static const OrtApi *
get_ort_api(void)
{
    if (g_ort == NULL) {
        const OrtApiBase *base = OrtGetApiBase();
        if (base != NULL) {
            g_ort = base->GetApi(ORT_API_VERSION);
        }
    }
    return g_ort;
}

/* --- Error helpers --- */

static void
set_error(char **out_error, const char *msg)
{
    if (out_error == NULL || msg == NULL) {
        return;
    }
    size_t len = strlen(msg);
    char *copy = (char *)malloc(len + 1);
    if (copy == NULL) {
        /* OOM while reporting an OOM — best we can do is leave
         * *out_error NULL and rely on the return code. */
        *out_error = NULL;
        return;
    }
    memcpy(copy, msg, len + 1);
    *out_error = copy;
}

/* Consume an OrtStatus*: extract code + message, release it, and
 * return the code. NULL status → ORT_OK with no message. The
 * caller's out_error receives a malloc'd copy of the message (or
 * stays NULL on success). */
static int
consume_status(OrtStatusPtr status, char **out_error)
{
    if (out_error != NULL) {
        *out_error = NULL;
    }
    if (status == NULL) {
        return ORT_OK;
    }
    const OrtApi *ort = get_ort_api();
    if (ort == NULL) {
        /* Can't extract a message without the API table; swallow
         * the status to avoid a leak and report a synthetic error. */
        set_error(out_error, "ONNX Runtime API unavailable");
        return ORT_FAIL;
    }
    OrtErrorCode code = ort->GetErrorCode(status);
    const char *msg = ort->GetErrorMessage(status);
    set_error(out_error, msg ? msg : "(no error message)");
    ort->ReleaseStatus(status);
    return (int)code;
}

static int
no_api_error(char **out_error)
{
    set_error(out_error,
              "ONNX Runtime API unavailable "
              "(OrtGetApiBase()->GetApi(ORT_API_VERSION) returned NULL). "
              "Shim compiled against ORT_API_VERSION "
#define STR_(x) #x
#define STR(x) STR_(x)
              STR(ORT_API_VERSION)
#undef STR
#undef STR_
              ".");
    return ORT_FAIL;
}

void
onnx_shim_free_error(char *err)
{
    free(err);
}

int
onnx_shim_api_version(void)
{
    return ORT_API_VERSION;
}

/* --- Env --- */

int
onnx_shim_init(const char *log_id, OrtEnv **out_env, char **out_error)
{
    if (out_error != NULL) { *out_error = NULL; }
    if (out_env == NULL) {
        set_error(out_error, "out_env must not be NULL");
        return ORT_INVALID_ARGUMENT;
    }
    *out_env = NULL;
    const OrtApi *ort = get_ort_api();
    if (ort == NULL) {
        return no_api_error(out_error);
    }
    OrtStatusPtr st = ort->CreateEnv(
        ORT_LOGGING_LEVEL_WARNING,
        log_id ? log_id : "onnx-native",
        out_env);
    return consume_status(st, out_error);
}

void
onnx_shim_release_env(OrtEnv *env)
{
    if (env == NULL) { return; }
    const OrtApi *ort = get_ort_api();
    if (ort == NULL) { return; }
    ort->ReleaseEnv(env);
}

/* --- Session options --- */

int
onnx_shim_create_session_options(OrtSessionOptions **out, char **out_error)
{
    if (out_error != NULL) { *out_error = NULL; }
    if (out == NULL) {
        set_error(out_error, "out must not be NULL");
        return ORT_INVALID_ARGUMENT;
    }
    *out = NULL;
    const OrtApi *ort = get_ort_api();
    if (ort == NULL) {
        return no_api_error(out_error);
    }
    OrtStatusPtr st = ort->CreateSessionOptions(out);
    if (st != NULL) {
        return consume_status(st, out_error);
    }
    /* Default to ORT_ENABLE_ALL — matches ORT's Python / C++
     * defaults. Callers that want a different level can add a
     * session-options knob later; not exposed in v0.1. */
    st = ort->SetSessionGraphOptimizationLevel(*out, ORT_ENABLE_ALL);
    if (st != NULL) {
        int code = consume_status(st, out_error);
        ort->ReleaseSessionOptions(*out);
        *out = NULL;
        return code;
    }
    return ORT_OK;
}

void
onnx_shim_release_session_options(OrtSessionOptions *options)
{
    if (options == NULL) { return; }
    const OrtApi *ort = get_ort_api();
    if (ort == NULL) { return; }
    ort->ReleaseSessionOptions(options);
}

/* --- Provider registration --- */

/* Forward declarations for the append-provider free functions we
 * may or may not have depending on platform + build variant. The
 * CUDA / DirectML ones are referenced only inside their own
 * #ifdef blocks below, so missing symbols don't break the link. */
#ifdef ONNX_SHIM_WITH_CUDA
extern OrtStatusPtr ORT_API_CALL
OrtSessionOptionsAppendExecutionProvider_CUDA(
    OrtSessionOptions *options, int device_id);
#endif

#ifdef ONNX_SHIM_WITH_DML
extern OrtStatusPtr ORT_API_CALL
OrtSessionOptionsAppendExecutionProvider_DML(
    OrtSessionOptions *options, int device_id);
#endif

int
onnx_shim_enable_provider(OrtSessionOptions *options,
                          const char *provider_name,
                          int64_t flags,
                          char **out_error)
{
    if (out_error != NULL) { *out_error = NULL; }
    if (options == NULL || provider_name == NULL) {
        set_error(out_error, "options and provider_name must not be NULL");
        return ORT_INVALID_ARGUMENT;
    }
    if (strcmp(provider_name, "cpu") == 0) {
        /* CPU is always registered by default; nothing to do. */
        return ORT_OK;
    }
    if (strcmp(provider_name, "coreml") == 0) {
#ifdef __APPLE__
        OrtStatusPtr st = OrtSessionOptionsAppendExecutionProvider_CoreML(
            options, (uint32_t)flags);
        return consume_status(st, out_error);
#else
        set_error(out_error,
                  "CoreML execution provider is only available on Apple "
                  "platforms.");
        return ORT_NOT_IMPLEMENTED;
#endif
    }
    if (strcmp(provider_name, "cuda") == 0) {
#ifdef ONNX_SHIM_WITH_CUDA
        OrtStatusPtr st = OrtSessionOptionsAppendExecutionProvider_CUDA(
            options, (int)flags);
        return consume_status(st, out_error);
#else
        set_error(out_error,
                  "CUDA execution provider not compiled into this shim. "
                  "Reinstall ONNX::Native with ONNX_NATIVE_WITH_CUDA=1 and "
                  "the GPU prebuilt variant of ONNX Runtime.");
        return ORT_NOT_IMPLEMENTED;
#endif
    }
    if (strcmp(provider_name, "dml") == 0) {
#ifdef ONNX_SHIM_WITH_DML
        OrtStatusPtr st = OrtSessionOptionsAppendExecutionProvider_DML(
            options, (int)flags);
        return consume_status(st, out_error);
#else
        set_error(out_error,
                  "DirectML execution provider not compiled into this "
                  "shim. DirectML support is Windows-only and deferred to "
                  "a future release.");
        return ORT_NOT_IMPLEMENTED;
#endif
    }
    {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "Unknown execution provider: '%s'. Supported: cpu, coreml, "
                 "cuda, dml.",
                 provider_name);
        set_error(out_error, buf);
        return ORT_INVALID_ARGUMENT;
    }
}

/* --- Session --- */

int
onnx_shim_create_session_from_path(OrtEnv *env,
                                   const char *model_path,
                                   OrtSessionOptions *options,
                                   OrtSession **out,
                                   char **out_error)
{
    if (out_error != NULL) { *out_error = NULL; }
    if (env == NULL || model_path == NULL || options == NULL || out == NULL) {
        set_error(out_error,
                  "env, model_path, options, and out must all be non-NULL");
        return ORT_INVALID_ARGUMENT;
    }
    *out = NULL;
    const OrtApi *ort = get_ort_api();
    if (ort == NULL) {
        return no_api_error(out_error);
    }
#ifdef _WIN32
    /* ORTCHAR_T is wchar_t on Windows. We accept UTF-8 char* at
     * the shim boundary so the Raku side doesn't have to know
     * about Windows path encodings, then widen via the Win32 API
     * before handing off to ORT. */
    int wlen = MultiByteToWideChar(CP_UTF8, 0, model_path, -1, NULL, 0);
    if (wlen <= 0) {
        set_error(out_error, "Failed to size UTF-8 path for widening");
        return ORT_INVALID_ARGUMENT;
    }
    wchar_t *wpath = (wchar_t *)malloc((size_t)wlen * sizeof(wchar_t));
    if (wpath == NULL) {
        set_error(out_error, "Out of memory widening model path");
        return ORT_FAIL;
    }
    int conv = MultiByteToWideChar(CP_UTF8, 0, model_path, -1, wpath, wlen);
    if (conv <= 0) {
        free(wpath);
        set_error(out_error, "Failed to convert UTF-8 path to wide-char");
        return ORT_INVALID_ARGUMENT;
    }
    OrtStatusPtr st = ort->CreateSession(env, wpath, options, out);
    free(wpath);
#else
    /* ORTCHAR_T is plain char on POSIX — pass straight through. */
    OrtStatusPtr st = ort->CreateSession(env, model_path, options, out);
#endif
    return consume_status(st, out_error);
}

int
onnx_shim_create_session_from_buffer(OrtEnv *env,
                                     const void *buf,
                                     size_t len,
                                     OrtSessionOptions *options,
                                     OrtSession **out,
                                     char **out_error)
{
    if (out_error != NULL) { *out_error = NULL; }
    if (env == NULL || buf == NULL || options == NULL || out == NULL) {
        set_error(out_error,
                  "env, buf, options, and out must all be non-NULL");
        return ORT_INVALID_ARGUMENT;
    }
    *out = NULL;
    const OrtApi *ort = get_ort_api();
    if (ort == NULL) {
        return no_api_error(out_error);
    }
    OrtStatusPtr st = ort->CreateSessionFromArray(env, buf, len, options, out);
    return consume_status(st, out_error);
}

void
onnx_shim_release_session(OrtSession *session)
{
    if (session == NULL) { return; }
    const OrtApi *ort = get_ort_api();
    if (ort == NULL) { return; }
    ort->ReleaseSession(session);
}

/* --- Session introspection --- */

int
onnx_shim_session_input_count(OrtSession *session,
                              size_t *out,
                              char **out_error)
{
    if (out_error != NULL) { *out_error = NULL; }
    if (session == NULL || out == NULL) {
        set_error(out_error, "session and out must be non-NULL");
        return ORT_INVALID_ARGUMENT;
    }
    const OrtApi *ort = get_ort_api();
    if (ort == NULL) {
        return no_api_error(out_error);
    }
    return consume_status(ort->SessionGetInputCount(session, out), out_error);
}

int
onnx_shim_session_output_count(OrtSession *session,
                               size_t *out,
                               char **out_error)
{
    if (out_error != NULL) { *out_error = NULL; }
    if (session == NULL || out == NULL) {
        set_error(out_error, "session and out must be non-NULL");
        return ORT_INVALID_ARGUMENT;
    }
    const OrtApi *ort = get_ort_api();
    if (ort == NULL) {
        return no_api_error(out_error);
    }
    return consume_status(ort->SessionGetOutputCount(session, out), out_error);
}

/* Shared helper: allocate the default allocator, call getter,
 * which writes the name into allocator-owned memory. We copy to
 * our own malloc so onnx_shim_free_name can use plain free(). */
static int
fetch_name_via(OrtStatusPtr (*getter)(const OrtSession *, size_t,
                                       OrtAllocator *, char **),
               OrtSession *session,
               size_t idx,
               char **out_name,
               char **out_error)
{
    if (out_error != NULL) { *out_error = NULL; }
    if (session == NULL || out_name == NULL) {
        set_error(out_error, "session and out_name must be non-NULL");
        return ORT_INVALID_ARGUMENT;
    }
    *out_name = NULL;
    const OrtApi *ort = get_ort_api();
    if (ort == NULL) {
        return no_api_error(out_error);
    }
    OrtAllocator *alloc = NULL;
    OrtStatusPtr st = ort->GetAllocatorWithDefaultOptions(&alloc);
    if (st != NULL) { return consume_status(st, out_error); }

    char *allocator_name = NULL;
    st = getter(session, idx, alloc, &allocator_name);
    if (st != NULL) { return consume_status(st, out_error); }

    /* Copy into malloc so the Raku side can release via plain
     * free() (via onnx_shim_free_name below). */
    size_t len = strlen(allocator_name);
    char *copy = (char *)malloc(len + 1);
    if (copy == NULL) {
        /* Best-effort cleanup on OOM — ignore AllocatorFree's
         * status since we're already in a failure path. */
        OrtStatusPtr free_st = ort->AllocatorFree(alloc, allocator_name);
        if (free_st != NULL) { ort->ReleaseStatus(free_st); }
        set_error(out_error, "Out of memory copying name");
        return ORT_FAIL;
    }
    memcpy(copy, allocator_name, len + 1);
    OrtStatusPtr free_st = ort->AllocatorFree(alloc, allocator_name);
    if (free_st != NULL) { ort->ReleaseStatus(free_st); }
    *out_name = copy;
    return ORT_OK;
}

int
onnx_shim_session_input_name(OrtSession *session,
                             size_t idx,
                             char **out_name,
                             char **out_error)
{
    const OrtApi *ort = get_ort_api();
    if (ort == NULL) { return no_api_error(out_error); }
    return fetch_name_via(ort->SessionGetInputName, session, idx,
                          out_name, out_error);
}

int
onnx_shim_session_output_name(OrtSession *session,
                              size_t idx,
                              char **out_name,
                              char **out_error)
{
    const OrtApi *ort = get_ort_api();
    if (ort == NULL) { return no_api_error(out_error); }
    return fetch_name_via(ort->SessionGetOutputName, session, idx,
                          out_name, out_error);
}

void
onnx_shim_free_name(char *name)
{
    /* We allocated via malloc in fetch_name_via, so plain free. */
    free(name);
}

/* Shared helper: given an OrtTensorTypeAndShapeInfo*, populate
 * out_elem_type, out_rank, and up to shape_cap dims into
 * out_shape. The info pointer is NOT released here — callers
 * handle that. */
static int
populate_tensor_info(const OrtTensorTypeAndShapeInfo *info,
                     int32_t *out_elem_type,
                     size_t *out_rank,
                     int64_t *out_shape,
                     size_t shape_cap,
                     char **out_error)
{
    const OrtApi *ort = get_ort_api();
    ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    OrtStatusPtr st = ort->GetTensorElementType(info, &elem_type);
    if (st != NULL) { return consume_status(st, out_error); }
    if (out_elem_type != NULL) { *out_elem_type = (int32_t)elem_type; }

    size_t rank = 0;
    st = ort->GetDimensionsCount(info, &rank);
    if (st != NULL) { return consume_status(st, out_error); }
    if (out_rank != NULL) { *out_rank = rank; }

    if (out_shape != NULL && shape_cap > 0 && rank > 0) {
        size_t to_read = rank < shape_cap ? rank : shape_cap;
        st = ort->GetDimensions(info, out_shape, to_read);
        if (st != NULL) { return consume_status(st, out_error); }
    }
    return ORT_OK;
}

static int
session_type_info_common(OrtStatusPtr (*getter)(const OrtSession *, size_t,
                                                 OrtTypeInfo **),
                         OrtSession *session,
                         size_t idx,
                         int32_t *out_elem_type,
                         size_t *out_rank,
                         int64_t *out_shape,
                         size_t shape_cap,
                         char **out_error)
{
    if (out_error != NULL) { *out_error = NULL; }
    if (session == NULL) {
        set_error(out_error, "session must be non-NULL");
        return ORT_INVALID_ARGUMENT;
    }
    const OrtApi *ort = get_ort_api();
    if (ort == NULL) { return no_api_error(out_error); }

    OrtTypeInfo *type_info = NULL;
    OrtStatusPtr st = getter(session, idx, &type_info);
    if (st != NULL) { return consume_status(st, out_error); }

    const OrtTensorTypeAndShapeInfo *tinfo = NULL;
    st = ort->CastTypeInfoToTensorInfo(type_info, &tinfo);
    if (st != NULL) {
        int code = consume_status(st, out_error);
        ort->ReleaseTypeInfo(type_info);
        return code;
    }
    if (tinfo == NULL) {
        ort->ReleaseTypeInfo(type_info);
        set_error(out_error,
                  "Input/output is not a tensor type (sequence / map types "
                  "are not supported in ONNX::Native v0.1).");
        return ORT_NOT_IMPLEMENTED;
    }

    int code = populate_tensor_info(tinfo, out_elem_type, out_rank,
                                    out_shape, shape_cap, out_error);
    ort->ReleaseTypeInfo(type_info);
    return code;
}

int
onnx_shim_session_input_type_info(OrtSession *session,
                                  size_t idx,
                                  int32_t *out_elem_type,
                                  size_t *out_rank,
                                  int64_t *out_shape,
                                  size_t shape_cap,
                                  char **out_error)
{
    const OrtApi *ort = get_ort_api();
    if (ort == NULL) { return no_api_error(out_error); }
    return session_type_info_common(ort->SessionGetInputTypeInfo,
                                    session, idx, out_elem_type,
                                    out_rank, out_shape, shape_cap,
                                    out_error);
}

int
onnx_shim_session_output_type_info(OrtSession *session,
                                   size_t idx,
                                   int32_t *out_elem_type,
                                   size_t *out_rank,
                                   int64_t *out_shape,
                                   size_t shape_cap,
                                   char **out_error)
{
    const OrtApi *ort = get_ort_api();
    if (ort == NULL) { return no_api_error(out_error); }
    return session_type_info_common(ort->SessionGetOutputTypeInfo,
                                    session, idx, out_elem_type,
                                    out_rank, out_shape, shape_cap,
                                    out_error);
}

/* --- Tensors --- */

int
onnx_shim_create_tensor(const void *data,
                        size_t byte_len,
                        const int64_t *shape,
                        size_t rank,
                        int32_t elem_type,
                        OrtValue **out,
                        char **out_error)
{
    if (out_error != NULL) { *out_error = NULL; }
    if (data == NULL || shape == NULL || out == NULL) {
        set_error(out_error, "data, shape, and out must all be non-NULL");
        return ORT_INVALID_ARGUMENT;
    }
    *out = NULL;
    const OrtApi *ort = get_ort_api();
    if (ort == NULL) { return no_api_error(out_error); }

    OrtMemoryInfo *mem_info = NULL;
    OrtStatusPtr st = ort->CreateCpuMemoryInfo(
        OrtDeviceAllocator, OrtMemTypeDefault, &mem_info);
    if (st != NULL) { return consume_status(st, out_error); }

    /* CreateTensorWithDataAsOrtValue does NOT copy data; the
     * OrtValue borrows the pointer. The Raku Tensor class is
     * responsible for keeping the backing Blob alive for the
     * OrtValue's lifetime. */
    st = ort->CreateTensorWithDataAsOrtValue(
        mem_info,
        (void *)data,   /* ORT signature takes mutable; Raku passes
                         * Blob-backed memory which is read-only to
                         * ORT during inference. */
        byte_len,
        shape,
        rank,
        (ONNXTensorElementDataType)elem_type,
        out);
    int code = consume_status(st, out_error);
    ort->ReleaseMemoryInfo(mem_info);
    return code;
}

void
onnx_shim_release_value(OrtValue *value)
{
    if (value == NULL) { return; }
    const OrtApi *ort = get_ort_api();
    if (ort == NULL) { return; }
    ort->ReleaseValue(value);
}

int
onnx_shim_tensor_shape(OrtValue *value,
                       int32_t *out_elem_type,
                       size_t *out_rank,
                       int64_t *out_shape,
                       size_t shape_cap,
                       char **out_error)
{
    if (out_error != NULL) { *out_error = NULL; }
    if (value == NULL) {
        set_error(out_error, "value must be non-NULL");
        return ORT_INVALID_ARGUMENT;
    }
    const OrtApi *ort = get_ort_api();
    if (ort == NULL) { return no_api_error(out_error); }

    OrtTensorTypeAndShapeInfo *tinfo = NULL;
    OrtStatusPtr st = ort->GetTensorTypeAndShape(value, &tinfo);
    if (st != NULL) { return consume_status(st, out_error); }

    int code = populate_tensor_info(tinfo, out_elem_type, out_rank,
                                    out_shape, shape_cap, out_error);
    ort->ReleaseTensorTypeAndShapeInfo(tinfo);
    return code;
}

int
onnx_shim_tensor_data(OrtValue *value,
                      void **out_data,
                      size_t *out_byte_len,
                      char **out_error)
{
    if (out_error != NULL) { *out_error = NULL; }
    if (value == NULL || out_data == NULL) {
        set_error(out_error, "value and out_data must be non-NULL");
        return ORT_INVALID_ARGUMENT;
    }
    *out_data = NULL;
    const OrtApi *ort = get_ort_api();
    if (ort == NULL) { return no_api_error(out_error); }

    OrtStatusPtr st = ort->GetTensorMutableData(value, out_data);
    if (st != NULL) { return consume_status(st, out_error); }

    if (out_byte_len != NULL) {
        /* Derive byte length from shape + element size. ORT
         * doesn't expose this directly — we compute it the same
         * way ORT does internally. */
        OrtTensorTypeAndShapeInfo *tinfo = NULL;
        st = ort->GetTensorTypeAndShape(value, &tinfo);
        if (st != NULL) { return consume_status(st, out_error); }

        size_t elem_count = 0;
        st = ort->GetTensorShapeElementCount(tinfo, &elem_count);
        if (st != NULL) {
            int code = consume_status(st, out_error);
            ort->ReleaseTensorTypeAndShapeInfo(tinfo);
            return code;
        }

        ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        st = ort->GetTensorElementType(tinfo, &elem_type);
        ort->ReleaseTensorTypeAndShapeInfo(tinfo);
        if (st != NULL) { return consume_status(st, out_error); }

        size_t elem_size = 0;
        switch (elem_type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:    elem_size = 4; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:   elem_size = 8; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:     elem_size = 1; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:    elem_size = 1; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:    elem_size = 2; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:   elem_size = 2; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:    elem_size = 4; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:   elem_size = 4; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:    elem_size = 8; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:   elem_size = 8; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:     elem_size = 1; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:  elem_size = 2; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: elem_size = 2; break;
            default:
                set_error(out_error,
                          "Tensor element type not supported in "
                          "ONNX::Native v0.1 for byte-length queries "
                          "(string / complex / int4 / fp8).");
                return ORT_NOT_IMPLEMENTED;
        }
        *out_byte_len = elem_count * elem_size;
    }
    return ORT_OK;
}

/* --- Run --- */

int
onnx_shim_run(OrtSession *session,
              const char *const *input_names,
              OrtValue *const *inputs,
              size_t num_inputs,
              const char *const *output_names,
              OrtValue **outputs,
              size_t num_outputs,
              char **out_error)
{
    if (out_error != NULL) { *out_error = NULL; }
    if (session == NULL) {
        set_error(out_error, "session must be non-NULL");
        return ORT_INVALID_ARGUMENT;
    }
    if (num_inputs > 0 && (input_names == NULL || inputs == NULL)) {
        set_error(out_error,
                  "input_names and inputs must be non-NULL when "
                  "num_inputs > 0");
        return ORT_INVALID_ARGUMENT;
    }
    if (num_outputs > 0 && (output_names == NULL || outputs == NULL)) {
        set_error(out_error,
                  "output_names and outputs must be non-NULL when "
                  "num_outputs > 0");
        return ORT_INVALID_ARGUMENT;
    }
    const OrtApi *ort = get_ort_api();
    if (ort == NULL) { return no_api_error(out_error); }

    /* Pre-zero outputs so partial failures don't leave us with
     * mystery pointers the caller might think are valid. */
    for (size_t i = 0; i < num_outputs; ++i) {
        outputs[i] = NULL;
    }

    /* ORT's signature takes OrtValue* const* for inputs, not
     * const OrtValue* const* — cast is safe (we're not modifying
     * the pointed-to values). */
    OrtStatusPtr st = ort->Run(
        session,
        NULL,                           /* run_options: use defaults */
        input_names,
        (const OrtValue *const *)inputs,
        num_inputs,
        output_names,
        num_outputs,
        outputs);
    return consume_status(st, out_error);
}

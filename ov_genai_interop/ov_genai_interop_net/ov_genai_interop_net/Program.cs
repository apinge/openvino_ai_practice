using System;
using System.Runtime.InteropServices;
using static NativeMethods;
using static StreamerCallback;

public enum ov_status_e
{
    OK = 0,
    GENERAL_ERROR = -1,
    NOT_IMPLEMENTED = -2,
    NETWORK_NOT_LOADED = -3,
    PARAMETER_MISMATCH = -4,
    NOT_FOUND = -5,
    OUT_OF_BOUNDS = -6,
    UNEXPECTED = -7,
    REQUEST_BUSY = -8,
    RESULT_NOT_READY = -9,
    NOT_ALLOCATED = -10,
    INFER_NOT_STARTED = -11,
    NETWORK_NOT_READ = -12,
    INFER_CANCELLED = -13,
    INVALID_C_PARAM = -14,
    UNKNOWN_C_ERROR = -15,
    NOT_IMPLEMENT_C_METHOD = -16,
    UNKNOW_EXCEPTION = -17
}

public struct streamer_callback
{
    public IntPtr callback_func;  // Pointer to the callback function
    public IntPtr args;  // Callback arguments
}

public static class NativeMethods
{
    [DllImport("openvino_genai_c.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern ov_status_e ov_genai_llm_pipeline_create(
        [MarshalAs(UnmanagedType.LPStr)] string models_path,
        [MarshalAs(UnmanagedType.LPStr)] string device,
        out IntPtr pipe);

    [DllImport("openvino_genai_c.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void ov_genai_llm_pipeline_free(IntPtr pipeline);

    [DllImport("openvino_genai_c.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern ov_status_e ov_genai_llm_pipeline_start_chat(IntPtr pipe);

    [DllImport("openvino_genai_c.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern ov_status_e ov_genai_llm_pipeline_finish_chat(IntPtr pipe);

    // Generation Config
    [DllImport("openvino_genai_c.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern ov_status_e ov_genai_generation_config_create(out IntPtr config);

    [DllImport("openvino_genai_c.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void ov_genai_generation_config_free(IntPtr handle);

    [DllImport("openvino_genai_c.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern ov_status_e ov_genai_generation_config_set_max_new_tokens(
        IntPtr config,
        ulong value);

    // Create DecodedResults
    [DllImport("openvino_genai_c.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern ov_status_e ov_genai_decoded_results_create(out IntPtr results);

    // Free DecodedResults
    [DllImport("openvino_genai_c.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void ov_genai_decoded_results_free(IntPtr results);

    // Retrieve the string from DecodedResults
    [DllImport("openvino_genai_c.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern ov_status_e ov_genai_decoded_results_get_string(IntPtr results, IntPtr output, ref ulong output_size);

    // Allow streamer to be null in C#

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void MyCallbackDelegate(IntPtr something);

    [DllImport("openvino_genai_c.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern ov_status_e ov_genai_llm_pipeline_generate(
        IntPtr pipe,
        string inputs,
        IntPtr config,
        [MarshalAs(UnmanagedType.FunctionPtr)] MyCallbackDelegate? callback,
        out IntPtr decodedResultsPtr);
}

public class GenerationConfig : IDisposable
{
    private IntPtr _configPtr;

    public GenerationConfig()
    {
        var status = NativeMethods.ov_genai_generation_config_create(out _configPtr);
        if (status != ov_status_e.OK || _configPtr == IntPtr.Zero)
        {
            Console.WriteLine($"Error: {status} when creating generation config.");
            throw new Exception("Failed to create generation config.");
        }
    }

    public void Dispose()
    {
        if (_configPtr != IntPtr.Zero)
        {
            NativeMethods.ov_genai_generation_config_free(_configPtr);
            _configPtr = IntPtr.Zero;
        }

        GC.SuppressFinalize(this);
    }

    public void SetMaxNewTokens(ulong value)
    {
        var status = NativeMethods.ov_genai_generation_config_set_max_new_tokens(_configPtr, value);
        if (status != ov_status_e.OK)
        {
            Console.WriteLine($"Error: {status} when setting max new tokens.");
            throw new Exception("Failed to set max new tokens.");
        }

        Console.WriteLine($"Max new tokens set to {value}.");
    }

    public IntPtr GetNativePointer() => _configPtr;
}

public class DecodedResults : IDisposable
{
    private IntPtr _nativePtr;

    // Constructor 1: Manually create
    public DecodedResults()
    {
        var status = NativeMethods.ov_genai_decoded_results_create(out _nativePtr);
        if (status != ov_status_e.OK || _nativePtr == IntPtr.Zero)
        {
            throw new Exception("Failed to create decoded results.");
        }
    }

    // ✅ Constructor 2: Wrap from an existing pointer (used for llm_pipeline_generate)
    internal DecodedResults(IntPtr nativePtr)
    {
        _nativePtr = nativePtr;
    }

    public string GetDecodedString()
    {
        ulong outputSize = 1024;
        IntPtr outputPtr = Marshal.AllocHGlobal((int)outputSize);

        var status = NativeMethods.ov_genai_decoded_results_get_string(_nativePtr, outputPtr, ref outputSize);
        if (status != ov_status_e.OK)
        {
            Marshal.FreeHGlobal(outputPtr);
            throw new Exception("Failed to get decoded string.");
        }

        string result = Marshal.PtrToStringAnsi(outputPtr);
        Marshal.FreeHGlobal(outputPtr);
        return result;
    }

    public void Dispose()
    {
        if (_nativePtr != IntPtr.Zero)
        {
            NativeMethods.ov_genai_decoded_results_free(_nativePtr);
            _nativePtr = IntPtr.Zero;
        }

        GC.SuppressFinalize(this);
    }
}

public class StreamerCallback
{
    // Define callback delegate
    public delegate void CallbackDelegate(string str, IntPtr args);

    public CallbackDelegate CallbackMethod;

    public StreamerCallback(CallbackDelegate callback)
    {
        CallbackMethod = callback;
    }

    // A static method must be provided for the C interface to convert to a C callback function pointer
    public static void CallbackMethodWrapper(string str, IntPtr args)
    {
        var callbackDelegate = Marshal.GetDelegateForFunctionPointer<CallbackDelegate>(args);
        callbackDelegate?.Invoke(str, args);
    }
}

public class LlmPipeline : IDisposable
{
    private IntPtr _nativePtr;

    public LlmPipeline(string modelPath, string device)
    {
        // Call the native method to create the pipeline
        var status = NativeMethods.ov_genai_llm_pipeline_create(modelPath, device, out _nativePtr);
        if (_nativePtr == IntPtr.Zero || status != ov_status_e.OK)
        {
            Console.WriteLine($"Error: {status} when creating LLM pipeline.");
            throw new Exception("Failed to create LLM pipeline.");
        }

        Console.WriteLine("LLM pipeline created successfully!");
    }

    public void Dispose()
    {
        if (_nativePtr != IntPtr.Zero)
        {
            NativeMethods.ov_genai_llm_pipeline_free(_nativePtr);
            _nativePtr = IntPtr.Zero;
        }

        GC.SuppressFinalize(this);
    }

    public string Generate(string input, GenerationConfig config, StreamerCallback callback = null)
    {
        streamer_callback streamer = new streamer_callback();
        if (callback != null)
        {
            var funcPtr = callback.CallbackMethod?.Method?.MethodHandle.GetFunctionPointer();
            streamer.callback_func = funcPtr.HasValue ? (IntPtr)funcPtr.Value : IntPtr.Zero;
            streamer.args = IntPtr.Zero;
        }
        //Console.WriteLine($"Streamer callback function pointer: {streamer.callback_func}");
        IntPtr configPtr = config.GetNativePointer();

       // Console.WriteLine($"Config pointer: {configPtr}"); // Print config pointer
        // Callback function can be null
        MyCallbackDelegate? cb =  null;
        IntPtr decodedPtr;
        var status = NativeMethods.ov_genai_llm_pipeline_generate(
            _nativePtr,
            input,
            configPtr,
            cb,
           out decodedPtr
        );
        
        if (status != ov_status_e.OK)
        {
            Console.WriteLine($"Error: {status} during generation.");
            throw new Exception("Failed to generate results.");
        }

        using var decoded = new DecodedResults(decodedPtr);
        string result = decoded.GetDecodedString();
        return result;
    }


    public void StartChat()
    {
        var status = NativeMethods.ov_genai_llm_pipeline_start_chat(_nativePtr);
        if (status != ov_status_e.OK)
        {
            Console.WriteLine($"Error: {status} when starting chat.");
            throw new Exception("Failed to start chat.");
        }

        Console.WriteLine("Chat started successfully!");
    }

    public void FinishChat()
    {
        var status = NativeMethods.ov_genai_llm_pipeline_finish_chat(_nativePtr);
        if (status != ov_status_e.OK)
        {
            Console.WriteLine($"Error: {status} when finishing chat.");
            throw new Exception("Failed to finish chat.");
        }

        Console.WriteLine("Chat finished successfully!");
    }
}

class Program
{
    static void Main(string[] args)
    {
        string modelDir = "E:\\repos\\tiny-llama-1.1b-chat_OV_FP16-INT8_ASYM";
        try
        {
            using var pipeline = new LlmPipeline(modelDir, "CPU");
            Console.WriteLine("Pipeline created!");

            var decodedResults = new DecodedResults();
            var generationConfig = new GenerationConfig();
            generationConfig.SetMaxNewTokens(100);

            pipeline.StartChat();
            string result = pipeline.Generate("Hello, how are you?", generationConfig, null);

            Console.WriteLine($"Decoded result: {result}");

            pipeline.FinishChat();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred: {ex.Message}");
        }
    }
}

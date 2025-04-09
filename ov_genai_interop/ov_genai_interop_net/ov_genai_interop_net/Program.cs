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

public enum ov_genai_streamming_status_e
{
    OV_GENAI_STREAMMING_STATUS_RUNNING = 0,  // Continue to run inference
    OV_GENAI_STREAMMING_STATUS_STOP =
        1,  // Stop generation, keep history as is, KV cache includes last request and generated tokens
    OV_GENAI_STREAMMING_STATUS_CANCEL = 2  // Stop generate, drop last prompt and all generated tokens from history, KV
                                           // cache includes history but last step
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
    public delegate void MyCallbackDelegate(IntPtr str, IntPtr args);

    [DllImport("openvino_genai_c.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern ov_status_e ov_genai_llm_pipeline_generate(
    IntPtr pipe,
    string inputs,
    IntPtr config,
    IntPtr streamer,  // This argument can be null
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


    public DecodedResults()
    {
        var status = NativeMethods.ov_genai_decoded_results_create(out _nativePtr);
        if (status != ov_status_e.OK || _nativePtr == IntPtr.Zero)
        {
            throw new Exception("Failed to create decoded results.");
        }
    }

    // Wrap from an existing pointer (used for llm_pipeline_generate)
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
[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
public delegate ov_genai_streamming_status_e MyCallbackDelegate(IntPtr str, IntPtr args);

[StructLayout(LayoutKind.Sequential)]
public struct streamer_callback
{
    public MyCallbackDelegate callback_func;
    public IntPtr args;
}
public class StreamerCallback : IDisposable
{
    public Action<string> OnStream;
    public MyCallbackDelegate Delegate;
    private GCHandle _selfHandle;

    public StreamerCallback(Action<string> onStream)
    {
        OnStream = onStream;
        Delegate = new MyCallbackDelegate(CallbackWrapper);
        _selfHandle = GCHandle.Alloc(this); 
    }

    public IntPtr ToNativePtr()
    {
        var native = new streamer_callback
        {
            callback_func = Delegate,
            args = GCHandle.ToIntPtr(_selfHandle)
        };

        IntPtr ptr = Marshal.AllocHGlobal(Marshal.SizeOf<streamer_callback>());
        Marshal.StructureToPtr(native, ptr, false);
        return ptr;
    }

    public void Dispose()
    {
        if (_selfHandle.IsAllocated)
            _selfHandle.Free();
    }

    private ov_genai_streamming_status_e CallbackWrapper(IntPtr str, IntPtr args)
    {
        string content = Marshal.PtrToStringAnsi(str) ?? string.Empty;

        if (args != IntPtr.Zero)
        {
            var handle = GCHandle.FromIntPtr(args);
            if (handle.Target is StreamerCallback self)
            {
                self.OnStream?.Invoke(content);
            }
        }

        return ov_genai_streamming_status_e.OV_GENAI_STREAMMING_STATUS_RUNNING;
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

    public string Generate(string input, GenerationConfig config, StreamerCallback? callback = null)
    {
        IntPtr configPtr = config.GetNativePointer();
        IntPtr decodedPtr;

        IntPtr streamerPtr = IntPtr.Zero;

        if (callback != null)
        {
            streamerPtr = callback.ToNativePtr(); 
        }

        var status = NativeMethods.ov_genai_llm_pipeline_generate(
            _nativePtr,
            input,
            configPtr,
            streamerPtr,  // This cargument can be null
            out decodedPtr
        );

        if (streamerPtr != IntPtr.Zero)
            Marshal.FreeHGlobal(streamerPtr); 

        callback?.Dispose();  // 👈 释放 GCHandle

        if (status != ov_status_e.OK)
        {
            Console.WriteLine($"Error: {status} during generation.");
            throw new Exception("Failed to generate results.");
        }

        using var decoded = new DecodedResults(decodedPtr);
        return decoded.GetDecodedString();
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
        if (args.Length < 1)
        {
            Console.WriteLine("Usage: your_app.exe <model_dir> ");
            return;
        }
        string modelDir = args[0];
        try
        {
            using var pipeline = new LlmPipeline(modelDir, "CPU");
            Console.WriteLine("Pipeline created!");

            var generationConfig = new GenerationConfig();
            generationConfig.SetMaxNewTokens(100); 
            pipeline.StartChat();  

            Console.WriteLine("question:");
            while (true)
            {
                string? input = Console.ReadLine();
                if (string.IsNullOrWhiteSpace(input)) break; 

                using var streamerCallback = new StreamerCallback((string chunk) =>
                {
                    Console.Write(chunk); 
                });

                string result = pipeline.Generate(input, generationConfig, streamerCallback);

                Console.WriteLine("\n----------\nquestion:");
            }

            pipeline.FinishChat();  
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred: {ex.Message}");
        }
    }


}

import Foundation
import llama

enum LlamaError: Error {
    case couldNotInitializeContext
}

func llama_batch_clear(_ batch: inout llama_batch) {
    batch.n_tokens = 0
}

func llama_batch_add(_ batch: inout llama_batch, _ id: llama_token, _ pos: llama_pos, _ seq_ids: [llama_seq_id], _ logits: Bool) {
    batch.token   [Int(batch.n_tokens)] = id
    batch.pos     [Int(batch.n_tokens)] = pos
    batch.n_seq_id[Int(batch.n_tokens)] = Int32(seq_ids.count)
    for i in 0..<seq_ids.count {
        batch.seq_id[Int(batch.n_tokens)]![Int(i)] = seq_ids[i]
    }
    batch.logits  [Int(batch.n_tokens)] = logits ? 1 : 0

    batch.n_tokens += 1
}

struct LlamaParams {
    var nPredict: Int = 256
    var nBatch: Int = 2048
}

actor LlamaContext {
    private var model: OpaquePointer
    private var context: OpaquePointer
    private var batch: llama_batch
    private var tokens_list: [llama_token]
    private var clip_ctx: OpaquePointer?
    private var systemPrompt: String
    private var userPromptPostfix: String
    private var sampling_ctx: SamplingWrapper
    private var n_past: Int32 = 0
    private var needsSystemInit = true

    /// This variable is used to store temporarily invalid cchars
    private var temporary_invalid_cchars: [CChar]

    var n_len: Int32 = 2048
    var n_cur: Int32 = 0

    var n_decode: Int32 = 0

    init(model: OpaquePointer, context: OpaquePointer, clip_ctx: OpaquePointer?, systemPrompt: String, userPromptPostfix: String) {
        self.model = model
        self.context = context
        self.clip_ctx = clip_ctx ?? nil
        self.tokens_list = []
        self.batch = llama_batch_init(512, 0, 1)
        self.temporary_invalid_cchars = []
        self.systemPrompt = systemPrompt
        self.userPromptPostfix = userPromptPostfix
        self.sampling_ctx = SamplingWrapper(llamaCtx: context)
        
    }

    deinit {
        if clip_ctx != nil {
            clip_free(clip_ctx)
        }
        llama_batch_free(batch)
        llama_free(context)
//        self.sampling_ctx.freeSamplingContext()
        llama_free_model(model)
        llama_backend_free()
    }

    static func create_context(path: String, clipPath: String?, systemPrompt: String, userPromptPostfix: String) throws -> LlamaContext {
        llama_backend_init()
        var model_params = llama_model_default_params()

#if targetEnvironment(simulator)
        model_params.n_gpu_layers = 0
        print("Running on simulator, force use n_gpu_layers = 0")
#endif
        let model = llama_load_model_from_file(path, model_params)
        guard let model else {
            print("Could not load model at \(path)")
            throw LlamaError.couldNotInitializeContext
        }
        let clip_ctx = clipPath.map { clipPath in
            clip_model_load(clipPath, 1)
        }

        let n_threads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        print("Using \(n_threads) threads")

        var ctx_params = llama_context_default_params()
        ctx_params.seed  = 1234
        ctx_params.n_ctx = 2048
        ctx_params.n_threads       = UInt32(n_threads)
        ctx_params.n_threads_batch = UInt32(n_threads)

        let context = llama_new_context_with_model(model, ctx_params)
        
        guard let context else {
            print("Could not load context!")
            throw LlamaError.couldNotInitializeContext
        }
        

        return LlamaContext(model: model, context: context, clip_ctx: clip_ctx ?? nil, systemPrompt: systemPrompt, userPromptPostfix: userPromptPostfix)
    }

    func model_info() -> String {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 256)
        result.initialize(repeating: Int8(0), count: 256)
        defer {
            result.deallocate()
        }

        // TODO: this is probably very stupid way to get the string from C

        let nChars = llama_model_desc(model, result, 256)
        let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nChars))

        var SwiftString = ""
        for char in bufferPointer {
            SwiftString.append(Character(UnicodeScalar(UInt8(char))))
        }

        return SwiftString
    }

    func get_n_tokens() -> Int32 {
        return batch.n_tokens;
    }
    
    func completion_system_init() {
        sampling_ctx.evaluateString(self.systemPrompt, batchSize: 2048, addBos: true)
        needsSystemInit = false
    }
   
    func completion_init(text: String, imageBytes: [UInt8]?) {
        print("attempting to complete \"\(text)\"")
        if needsSystemInit {
            completion_system_init()
        }
        if imageBytes != nil {
            var myBytes = imageBytes!;
            let size = Int32(myBytes.count)
            let success = myBytes.withUnsafeMutableBytes {raw in
                self.sampling_ctx.embedImage(raw.baseAddress, withSize: size, clipContext: clip_ctx)
            }
            
            print("prashanth: \(success)")
  
        }
        self.sampling_ctx.evaluateString("\(text)\(userPromptPostfix)", batchSize: 2048, addBos: false)
    }

    func completion_loop() -> String {
        needsSystemInit = true
        let ret = self.sampling_ctx.sampleAndEvaluate()!
        print(ret)
        n_cur += 1
        return ret

    }

    func bench(pp: Int, tg: Int, pl: Int, nr: Int = 1) -> String {
        var pp_avg: Double = 0
        var tg_avg: Double = 0

        var pp_std: Double = 0
        var tg_std: Double = 0

        for _ in 0..<nr {
            // bench prompt processing

            llama_batch_clear(&batch)

            let n_tokens = pp

            for i in 0..<n_tokens {
                llama_batch_add(&batch, 0, Int32(i), [0], false)
            }
            batch.logits[Int(batch.n_tokens) - 1] = 1 // true

            llama_kv_cache_clear(context)

            let t_pp_start = ggml_time_us()

            if llama_decode(context, batch) != 0 {
                print("llama_decode() failed during prompt")
            }

            let t_pp_end = ggml_time_us()

            // bench text generation

            llama_kv_cache_clear(context)

            let t_tg_start = ggml_time_us()

            for i in 0..<tg {
                llama_batch_clear(&batch)

                for j in 0..<pl {
                    llama_batch_add(&batch, 0, Int32(i), [Int32(j)], true)
                }

                if llama_decode(context, batch) != 0 {
                    print("llama_decode() failed during text generation")
                }
            }

            let t_tg_end = ggml_time_us()

            llama_kv_cache_clear(context)

            let t_pp = Double(t_pp_end - t_pp_start) / 1000000.0
            let t_tg = Double(t_tg_end - t_tg_start) / 1000000.0

            let speed_pp = Double(pp)    / t_pp
            let speed_tg = Double(pl*tg) / t_tg

            pp_avg += speed_pp
            tg_avg += speed_tg

            pp_std += speed_pp * speed_pp
            tg_std += speed_tg * speed_tg

            print("pp \(speed_pp) t/s, tg \(speed_tg) t/s")
        }

        pp_avg /= Double(nr)
        tg_avg /= Double(nr)

        if nr > 1 {
            pp_std = sqrt(pp_std / Double(nr - 1) - pp_avg * pp_avg * Double(nr) / Double(nr - 1))
            tg_std = sqrt(tg_std / Double(nr - 1) - tg_avg * tg_avg * Double(nr) / Double(nr - 1))
        } else {
            pp_std = 0
            tg_std = 0
        }

        let model_desc     = model_info();
        let model_size     = String(format: "%.2f GiB", Double(llama_model_size(model)) / 1024.0 / 1024.0 / 1024.0);
        let model_n_params = String(format: "%.2f B", Double(llama_model_n_params(model)) / 1e9);
        let backend        = "Metal";
        let pp_avg_str     = String(format: "%.2f", pp_avg);
        let tg_avg_str     = String(format: "%.2f", tg_avg);
        let pp_std_str     = String(format: "%.2f", pp_std);
        let tg_std_str     = String(format: "%.2f", tg_std);

        var result = ""

        result += String("| model | size | params | backend | test | t/s |\n")
        result += String("| --- | --- | --- | --- | --- | --- |\n")
        result += String("| \(model_desc) | \(model_size) | \(model_n_params) | \(backend) | pp \(pp) | \(pp_avg_str) ± \(pp_std_str) |\n")
        result += String("| \(model_desc) | \(model_size) | \(model_n_params) | \(backend) | tg \(tg) | \(tg_avg_str) ± \(tg_std_str) |\n")

        return result;
    }

    func clear() {
        tokens_list.removeAll()
        temporary_invalid_cchars.removeAll()
        llama_kv_cache_clear(context)
        sampling_ctx.resetSamplingContext()
    }
}

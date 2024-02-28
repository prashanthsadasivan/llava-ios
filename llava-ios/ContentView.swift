//
//  ContentView.swift
//  llava-ios
//
//  Created by Prashanth Sadasivan on 2/2/24.
//

import SwiftUI


struct LlamaModel: Identifiable{
    var id = UUID()
    var name: String
    var status: String
    var filename: String
    var url: String
}

class AppState: ObservableObject {
    
    let NS_PER_S = 1_000_000_000.0
    enum StartupState{
        case Startup
        case Loading
        case Started
    }
    
    var selectedBaseModel: LlamaModel?
    @Published var downloadedBaseModels: [LlamaModel] = []
    @Published var state: StartupState = .Startup
    @Published var useTiny: Bool = true
    @Published var messageLog = ""
    
    private var llamaContext: LlamaContext?
    
    func setBaseModel(model: LlamaModel?) {
        selectedBaseModel = model
    }
    
    public static func previewState() -> AppState {
        let ret = AppState()
        ret.downloadedBaseModels = [
            LlamaModel(name: "TinyLlama-1.1B Chat (Q8_0, 1.1 GiB)", status: "downloaded", filename: "tinyllama-1.1b-chat-v1.0.Q8_0.gguf", url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf?download=true"),
            LlamaModel(name: "mmproj model f16 (for Images)", status: "downloaded", filename: "mmproj-model-f16.gguf", url: "https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/mmproj-model-f16.gguf?download=true")
        ]
        return ret
    }
    
    func appendMessage(result: String) {
        DispatchQueue.main.async {
            self.messageLog += "\(result)"
        }
    }
    
    var TINY_SYS_PROMPT: String = "<|system|>\nYou are a chatbot who tries to answer questions that might refer to a picture the user provides!</s>\n<|user|>"
    
    var TINY_USER_POSTFIX: String = "</s>\n<|assistant|>"
    
    var DEFAULT_SYS_PROMPT: String = "USER:"
    var DEFAULT_USER_POSTFIX: String = "\nASSISTANT:"
   
    func complete(text: String, img: ProfileModel.ImageState?) async {
        if llamaContext == nil {
            print("loading modle: use tiny: \(useTiny)")
            let mmproj = downloadedBaseModels.first(where: {m in m.name.contains("mmproj")})
            let llava = downloadedBaseModels.first(where: {m in m.name.lowercased().contains("llava")})
            let tiny = downloadedBaseModels.first(where: {m in m.name.lowercased().contains("tiny")})
            let model = useTiny ? tiny : llava
            let systemPrompt = useTiny ? TINY_SYS_PROMPT : DEFAULT_SYS_PROMPT
            let userPostfix = useTiny ? TINY_USER_POSTFIX : DEFAULT_USER_POSTFIX
            if model != nil && mmproj != nil {
                print("GOT THE MODELS")
                do {
                    self.llamaContext = try LlamaContext.create_context(path: model!.filename, clipPath: mmproj!.filename, systemPrompt: systemPrompt, userPromptPostfix: userPostfix)
                } catch {
                    messageLog += "Error!\n"
                    return
                }
            } else {
                print("MISSING MODELS")
            }
        }
        guard let llamaContext else {
            return
        }
        let image: Data? = switch img {
        case .success(_, let d):
            d
        default:
            nil
        }
        let bytes = image.map { d in
            var byteArray = [UInt8](repeating: 0, count: d.count) // Create an array of the correct size
            d.copyBytes(to: &byteArray, count: d.count)
            return byteArray
        }
        
        let t_start = DispatchTime.now().uptimeNanoseconds
        await llamaContext.completion_init(text: text, imageBytes: bytes)
        let t_heat_end = DispatchTime.now().uptimeNanoseconds
        let t_heat = Double(t_heat_end - t_start) / NS_PER_S
        appendMessage(result: "\(text)")
        
        while await llamaContext.n_cur < llamaContext.n_len {
            let result = await llamaContext.completion_loop()
            appendMessage(result: "\(result)")
        }
        
        let t_end = DispatchTime.now().uptimeNanoseconds
        let t_generation = Double(t_end - t_heat_end) / NS_PER_S
        let tokens_per_second = Double(await llamaContext.n_len) / t_generation
        
        await llamaContext.clear()
        appendMessage(result: """
            \n
            Done
            Heat up took \(t_heat)s
            Generated \(tokens_per_second) t/s\n
            """
                      )
    }
    
    func clear() async {
        guard let llamaContext else {
            return
        }

        await llamaContext.clear()
        messageLog = ""
    }

    public func loadModelsFromDisk() {
        do {
            let documentsURL = getDocumentsDirectory()
            let modelURLs = try FileManager.default.contentsOfDirectory(at: documentsURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants])
            for modelURL in modelURLs {
                let modelName = modelURL.deletingPathExtension().lastPathComponent
                downloadedBaseModels.append(LlamaModel(name: modelName, status: "downloaded", filename: modelURL.path(), url: "-"))
            }

            state = .Started
        } catch {
            print("Error loading models from disk: \(error)")
        }
    }
    
    
    func getDocumentsDirectory() -> URL {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        return paths[0]
    }
}


struct ContentView: View {
    @StateObject var appstate = AppState()
    @State private var profileModel = ProfileModel()
    var body: some View {
        VStack {
            if appstate.state == .Startup {
                Text("Loading....")
            } else if appstate.state == .Started {
                if (appstate.downloadedBaseModels.count != 3) {
                    DownloadButton(appState: appstate, modelName: "TinyLlama-1.1B Chat (Q8_0, 1.1 GiB)", modelUrl: "https://huggingface.co/cmp-nct/llava-1.6-gguf/resolve/main/vicuna-7b-q5_k.gguf?download=true", filename: "tinyllama-1.1b-chat-v1.0.Q8_0.gguf")
                    
                    DownloadButton(appState: appstate, modelName: "LLaVa-v1.6-vicuna-7b (Q4_k_M, 4.08 GiB)", modelUrl: "https://huggingface.co/cmp-nct/llava-1.6-gguf/resolve/main/vicuna-7b-q5_k.gguf?download=true", filename: "llava-v1.6-vicuna-7B.Q4_k.gguf")
                    DownloadButton(appState: appstate, modelName: "mmproj model llava 1.6 f16 (for Images)", modelUrl: "https://huggingface.co/cmp-nct/llava-1.6-gguf/resolve/main/mmproj-vicuna7b-f16.gguf?download=true", filename: "mmproj-model-1.6-vicuna-f16.gguf")
                } else {
                    InferenceScreenView(appstate: appstate, profileModel: profileModel)
                }
            }
        }
        .padding()
        .onAppear {
            appstate.state = .Loading
            appstate.loadModelsFromDisk()
        }
    }
    
}

//#Preview {
//    ContentView()
//}

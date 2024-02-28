import SwiftUI
import Foundation
import Combine

struct DownloadButton: View {

    class DownloadState : ObservableObject {
        @Published var status: String = ""
        @Published  var progress = 0.0
        @Published var model: LlamaModel?
        var downloadTask: URLSessionDownloadTask?
        var downloadLlavaTask: URLSessionDownloadTask?
        var progressObserver: AnyCancellable?

        static func withStatus(_ status: String) -> DownloadState {
            return DownloadState(status: status)
        }

        init(status: String) {
            self.status = status
        }

        static func getFileURL(filename: String) -> URL {
            FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent(filename)
        }

        func download(modelName: String, modelUrl: String, filename: String) {
            status = "downloading"
            print("Downloading model \(modelName) from \(modelUrl)")
            guard let url = URL(string: modelUrl) else { return }
            let fileURL = DownloadState.getFileURL(filename: filename)

            downloadTask = URLSession.shared.downloadTask(with: url) { temporaryURL, response, error in
                if  let error = error {
                    print("Error: \(error.localizedDescription)")
                    return
                }

                guard let response = response as? HTTPURLResponse, (200...299).contains(response.statusCode) else {
                    print("Server error!")
                    return
                }

                do {
                    if let temporaryURL = temporaryURL {
                        try FileManager.default.copyItem(at: temporaryURL, to: fileURL)
                        print("Writing to \(filename) completed")
                        DispatchQueue.main.async {
                            self.model =  LlamaModel(name: modelUrl, status: filename, filename: "downloaded", url: modelUrl)
                            self.status = "downloaded"
                        }
                    }
                } catch  {
                    // Handle the error locally, for example, by logging it or updating the state
                    print("Error occurred: \(error)")
                    // Optionally, update some state or perform other non-throwing actions
                }
            }
            self.progressObserver = downloadTask?.progress
                .publisher(for: \.fractionCompleted).receive(on: DispatchQueue.main)
                .sink { [weak self] fractionCompleted in
                    self?.progress = fractionCompleted
                }

            downloadTask?.resume()
            downloadLlavaTask?.resume()
        }
    }

    @StateObject var downloadState: DownloadState
    @ObservedObject private var appState: AppState
    private var modelName: String
    private var modelUrl: String
    private var filename: String

    init(appState: AppState, modelName: String, modelUrl: String, filename: String) {
        self.appState = appState
        self.modelName = modelName
        self.modelUrl = modelUrl
        self.filename = filename
        let fileURL = DownloadState.getFileURL(filename: filename)
        _downloadState = StateObject(wrappedValue: DownloadState.withStatus(FileManager.default.fileExists(atPath: fileURL.path) ? "downloaded" : "download"
))
    }

    var body: some View {
        VStack {
            if downloadState.status == "download" {
                Button(action: {
                    downloadState.download(modelName: modelName, modelUrl: modelUrl, filename: filename)
                }) {
                    Text("Download " + modelName)
                }
            } else if downloadState.status == "downloading" {
                Button(action: {
                    downloadState.downloadTask?.cancel()
                    downloadState.status = "download"
                }) {
                    Text("\(modelName) (Downloading \(Int(downloadState.progress * 100))%)")
                }
            } else if downloadState.status == "downloaded" {
                Button(action: {
                    let fileURL = DownloadState.getFileURL(filename: filename)
                    appState.setBaseModel(model: downloadState.model)
                }) {
                    Text("Load \(modelName)")
                }
            } else {
                Text("Unknown status: \(downloadState.status)")
            }
        }
        .onDisappear() {
            downloadState.downloadTask?.cancel()
        }
    }
}

// #Preview {
//    DownloadButton(
//        llamaState: LlamaState(),
//        modelName: "TheBloke / TinyLlama-1.1B-1T-OpenOrca-GGUF (Q4_0)",
//        modelUrl: "https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyllama-1.1b-1t-openorca.Q4_0.gguf?download=true",
//        filename: "tinyllama-1.1b-1t-openorca.Q4_0.gguf"
//    )
// }

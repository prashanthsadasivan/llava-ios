//
//  ModelSelectScreen.swift
//  llava-ios
//
//  Created by Prashanth Sadasivan on 2/12/24.
//

import SwiftUI

struct InferenceScreenView: View {
    @StateObject var appstate: AppState
    @State private var multiLineText = ""
    @State private var cameraModel: CameraDataModel
    @FocusState private var focused: Bool
    
    init(appstate: AppState, multiLineText: String = "", cameraModel: CameraDataModel) {
        self._appstate = StateObject(wrappedValue: appstate)
        self.multiLineText = multiLineText
        self.cameraModel = cameraModel
    }
    
    var body: some View {
        VStack {
            if (self.cameraModel.currentImageData != nil) {
                CameraView(model: self.cameraModel)
                    .overlay {
                        ScrollView(.vertical, showsIndicators: true) {
                            Text(appstate.messageLog)
                                .font(.system(size: 12))
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding()
                                .background {
                                    Color.gray.opacity(0.2)
                                }
                        }
                    }
//Z
            } else {
                CameraView(model: self.cameraModel)
            }
//Z                .overlay {
//Z                    ScrollView(.vertical, showsIndicators: true) {
//Z                        Text(appstate.messageLog)
//Z                            .font(.system(size: 12))
//Z                            .frame(maxWidth: .infinity, alignment: .leading)
//Z                            .padding()
//Z                            .onTapGesture {
//Z                                UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
//Z                            }.background {
//Z                                Color.gray.opacity(0.2)
//Z                            }
//Z                    }
//Z                }
            VStack {
                TextEditor(text: $multiLineText)
                    .focused($focused)
                    .frame(height: 80)
                    .padding()
                    .border(Color.gray, width: 0.5)
                
                HStack {
                    Button("Clear") {
                        clear()
                    }
                    Button("Copy") {
                        UIPasteboard.general.string = appstate.messageLog
                    }
                    Button("Send") {
                        focused = false
                        sendText()
                    }
                    
                }
                .buttonStyle(.bordered)
                .padding()
                .onAppear {
                    Task {
                       await appstate.preInit()
                    }
                }
            }
        }
    }
    
    func sendText() {
        Task {
            await appstate.complete(text: multiLineText, img: cameraModel.currentImageData)
            multiLineText = ""
        }
    }

    func clear() {
        Task {
            await appstate.clear()
            DispatchQueue.main.async {
                cameraModel.currentImageData = nil
            }
        }
    }
}

#Preview {
    InferenceScreenView(appstate: AppState.previewState(), cameraModel: CameraDataModel())
}

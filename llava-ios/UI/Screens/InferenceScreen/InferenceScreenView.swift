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
    @State private var profileModel: ProfileModel
    @FocusState private var focused: Bool
    
    init(appstate: AppState, multiLineText: String = "", profileModel: ProfileModel) {
        self._appstate = StateObject(wrappedValue: appstate)
        self.multiLineText = multiLineText
        self.profileModel = profileModel
    }
    
    var body: some View {
        VStack {
            ScrollView(.vertical, showsIndicators: true) {
                Text(appstate.messageLog)
                    .font(.system(size: 12))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .onTapGesture {
                        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                    }
            }
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
                    
                    EditableCircularProfileImage(viewModel: profileModel)
                }
                .buttonStyle(.bordered)
                .padding()
                if !focused {
                    Toggle(isOn: $appstate.useTiny) {
                        Text("Use Tiny?")
                    }.padding()
                }
            }
        }
    }
    
    func sendText() {
        Task {
            await appstate.complete(text: multiLineText, img: profileModel.imageState)
            multiLineText = ""
        }
    }

    func clear() {
        Task {
            await appstate.clear()
        }
    }
}

#Preview {
    InferenceScreenView(appstate: AppState.previewState(), profileModel: ProfileModel())
}

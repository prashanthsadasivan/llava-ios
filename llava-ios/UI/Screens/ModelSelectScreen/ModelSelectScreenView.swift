//
//  ModelSelectScreenView.swift
//  llava-ios
//
//  Created by Prashanth Sadasivan on 3/2/24.
//

import Foundation
import SwiftUI

struct ModelSelectScreenView: View {
    @StateObject var appstate: AppState
    @ObservedObject var models: LlavaModelInfoList
    
    
    @State private var multiSelection = Set<UUID>()
    
    var body: some View {
        List(models.models, selection: $multiSelection) { m in
            Text(m.modelName)
        }
    }
}


#Preview {
    ModelSelectScreenView(appstate: AppState.previewState(), models: AppState.llavaModels)
}


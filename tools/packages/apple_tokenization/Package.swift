// swift-tools-version:5.5

import PackageDescription

let package = Package(
    name: "Tokenizer",
    platforms: [
        .macOS(.v10_14),
    ],
    products: [
        .library(name: "Tokenizer", type: .dynamic, targets: ["Tokenizer"]),
    ],
    dependencies: [],
    targets: [
        .target(name: "Tokenizer", dependencies: []),
    ]
)

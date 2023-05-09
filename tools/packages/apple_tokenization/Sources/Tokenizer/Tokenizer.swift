import NaturalLanguage

func tokenizeSentence(_ sentence: String) -> [(String, Int, Int)] {
    let tokenizer = NLTokenizer(unit: .word)
    tokenizer.string = sentence

    var tokenizedText: [(String, Int, Int)] = []

    tokenizer.enumerateTokens(in: sentence.startIndex..<sentence.endIndex) { (tokenRange, _) -> Bool in
        let token = String(sentence[tokenRange])
        let startIndex = sentence.distance(from: sentence.startIndex, to: tokenRange.lowerBound)
        let endIndex = startIndex + sentence.distance(from: tokenRange.lowerBound, to: tokenRange.upperBound)

        tokenizedText.append((token, startIndex, endIndex))
        return true
    }

    return tokenizedText
}


@_cdecl("tokenizeSentenceWrapper")
public func tokenizeSentenceWrapper(_ input: UnsafePointer<CChar>, _ output: UnsafeMutablePointer<CChar>, _ outputSize: Int) {
    let sentence = String(cString: input)
    let tokenizedSentence = tokenizeSentence(sentence)

    var result: [[Any]] = []
    for (token, start, end) in tokenizedSentence {
        result.append([token, start, end])
    }

    let json = try! JSONSerialization.data(withJSONObject: result)
    let jsonString = String(data: json, encoding: .utf8)!

    let jsonStringCString = jsonString.cString(using: .utf8)!

    for i in 0..<(min(outputSize - 1, jsonStringCString.count)) {
        output[i] = jsonStringCString[i]
    }
    output[min(outputSize - 1, jsonStringCString.count)] = 0
}

func isAnagram(s: String, t: String) -> Bool {
    // Early exit if lengths differ
    if s.count != t.count { return false }

    var sDict: [Character: Int] = [:]
    var tDict: [Character: Int] = [:]

    for i in 0..<s.count {
        let sIndex = s.index(s.startIndex, offsetBy: i)
        let tIndex = t.index(t.startIndex, offsetBy: i)

        sDict[s[sIndex], default: 0] += 1
        tDict[t[tIndex], default: 0] += 1
    }

    return sDict == tDict
}

func twoSum(numbers: [Int], target: Int) -> [Int] {
    var dict: [Int: Int] = [:]

    for i in 0..<numbers.count {
        dict[numbers[i]] = i
    }

    for i in 0..<numbers.count {
        let diff = target - numbers[i]

        if let diffIndex = dict[diff], diffIndex != i {
            if i < diffIndex {
                return [i, diffIndex]
            } else {
                return [diffIndex, i]
            }
        }
    }

    return [] 
}



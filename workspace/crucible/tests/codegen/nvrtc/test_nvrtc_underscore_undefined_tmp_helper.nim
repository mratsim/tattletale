## Helper: Nim parser rejects `const _` as local declaration.
type X_marker = object
const _* = X_marker()

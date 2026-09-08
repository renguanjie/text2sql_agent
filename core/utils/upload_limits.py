"""上传文件的统一大小与编码校验。"""


class UploadValidationError(ValueError):
    """上传文件不符合安全限制。"""


def read_uploaded_text(uploaded_file, max_bytes: int) -> str:
    """在读取内容前后都检查大小，并仅接受 UTF-8 文本。"""
    declared_size = getattr(uploaded_file, "size", None)
    if declared_size is not None and declared_size > max_bytes:
        raise UploadValidationError(f"上传文件超过 {max_bytes // (1024 * 1024)} MB 限制")
    payload = uploaded_file.getvalue()
    if len(payload) > max_bytes:
        raise UploadValidationError(f"上传文件超过 {max_bytes // (1024 * 1024)} MB 限制")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UploadValidationError("文件必须使用 UTF-8 编码") from exc

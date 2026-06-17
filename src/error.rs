use thiserror::Error;

#[derive(Error, Debug)]
pub enum PointCloudError {
    #[error("IO错误: {0}")]
    IoError(#[from] std::io::Error),

    #[error("格式解析错误: {0}")]
    ParseError(String),

    #[error("不支持的文件格式: {0}")]
    UnsupportedFormat(String),

    #[error("点云数据为空")]
    EmptyPointCloud,

    #[error("法向量未计算")]
    NormalsNotComputed,

    #[error("配准失败: {0}")]
    RegistrationFailed(String),

    #[error("重建失败: {0}")]
    ReconstructionFailed(String),

    #[error("配置错误: {0}")]
    ConfigError(String),

    #[error("YAML解析错误: {0}")]
    YamlError(#[from] serde_yaml::Error),

    #[error("JSON序列化错误: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("参数错误: {0}")]
    InvalidParameter(String),

    #[error("算法错误: {0}")]
    AlgorithmError(String),
}

pub type Result<T> = std::result::Result<T, PointCloudError>;

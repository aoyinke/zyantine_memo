"""
OpenAI 兼容的 API 服务
为语音 RTC 交互项目提供标准化的 API 接口
"""
import time
import uuid
import asyncio
import json
import os
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from zyantine_facade import ZyantineFacade, create_zyantine


class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色: system, user, assistant")
    content: str = Field(..., description="消息内容")


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="zyantine-v1", description="模型名称")
    messages: List[ChatMessage] = Field(..., description="对话消息列表")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="采样温度")
    max_tokens: Optional[int] = Field(default=1000, ge=1, description="最大生成token数")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="核采样参数")
    stream: Optional[bool] = Field(default=False, description="是否流式输出")
    user: Optional[str] = Field(default=None, description="用户标识")


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class ErrorResponse(BaseModel):
    error: Dict[str, Any]


class ConnectionManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """接受连接"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        print(f"客户端 {client_id} 已连接，当前连接数: {len(self.active_connections)}")

    def disconnect(self, client_id: str):
        """断开连接"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            print(f"客户端 {client_id} 已断开，当前连接数: {len(self.active_connections)}")

    async def send_message(self, message: str, client_id: str):
        """发送消息给指定客户端"""
        if client_id not in self.active_connections:
            print(f"警告: 客户端 {client_id} 不在活跃连接列表中")
            return False
        
        websocket = self.active_connections[client_id]
        
        try:
            await websocket.send_text(message)
            return True
        except RuntimeError as e:
            if "close message has been sent" in str(e):
                print(f"警告: 客户端 {client_id} 连接已关闭，无法发送消息")
            else:
                print(f"发送消息失败 (client_id={client_id}): {e}")
            self.disconnect(client_id)
            return False
        except Exception as e:
            print(f"发送消息时发生意外错误 (client_id={client_id}): {e}")
            self.disconnect(client_id)
            return False

    async def broadcast(self, message: str):
        """广播消息给所有客户端"""
        for connection in self.active_connections.values():
            await connection.send_text(message)


class APIServer:
    """API 服务器"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000,
                 api_key: Optional[str] = None, session_id: str = "default",
                 config_file: str = "config/llm_config.json"):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.session_id = session_id
        self.config_file = config_file
        self.facade: Optional[ZyantineFacade] = None
        self.app: Optional[FastAPI] = None
        self.startup_time: Optional[int] = None
        self.connection_manager = ConnectionManager()

    async def startup(self):
        """启动时初始化"""
        print("正在初始化自衍体 AI 系统...")
        
        try:
            # 优先使用配置文件
            if os.path.exists(self.config_file):
                self.facade = create_zyantine(
                    config_file=self.config_file,
                    session_id=self.session_id
                )
            elif self.api_key:
                # 如果配置文件不存在但提供了api_key，则使用api_key
                self.facade = create_zyantine(
                    api_key=self.api_key,
                    session_id=self.session_id
                )
            else:
                # 否则使用默认方式创建
                self.facade = ZyantineFacade(session_id=self.session_id)
            
            self.startup_time = int(time.time())
            print(f"系统初始化完成，会话ID: {self.session_id}")
            
        except Exception as e:
            print(f"系统初始化失败: {e}")
            raise

    async def shutdown(self):
        """关闭时清理"""
        print("正在关闭系统...")
        if self.facade:
            self.facade.shutdown()
        print("系统已关闭")

    def create_app(self) -> FastAPI:
        """创建 FastAPI 应用"""
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self.startup()
            yield
            await self.shutdown()

        self.app = FastAPI(
            title="Zyantine AI API",
            description="自衍体 AI 系统 - OpenAI 兼容接口",
            version="1.0.0",
            lifespan=lifespan
        )

        self._register_routes()
        self._register_exception_handlers()

        return self.app

    def _register_routes(self):
        """注册路由"""

        @self.app.get("/health")
        async def health_check():
            """健康检查"""
            return {
                "status": "healthy",
                "service": "zyantine-ai",
                "session_id": self.session_id,
                "uptime": int(time.time()) - self.startup_time if self.startup_time else 0
            }

        @self.app.get("/v1/models", response_model=ModelsResponse)
        async def list_models():
            """获取可用模型列表"""
            return ModelsResponse(
                data=[
                    ModelInfo(
                        id="zyantine-v1",
                        created=self.startup_time or int(time.time()),
                        owned_by="zyantine"
                    ),
                    ModelInfo(
                        id="zyantine-enhanced",
                        created=self.startup_time or int(time.time()),
                        owned_by="zyantine"
                    )
                ]
            )

        @self.app.get("/v1/models/{model_id}")
        async def get_model(model_id: str):
            """获取指定模型信息"""
            available_models = ["zyantine-v1", "zyantine-enhanced"]
            if model_id not in available_models:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{model_id}' not found"
                )
            return ModelInfo(
                id=model_id,
                created=self.startup_time or int(time.time()),
                owned_by="zyantine"
            )

        @self.app.post("/v1/chat/completions")
        async def create_chat_completion(request: ChatCompletionRequest):
            """创建聊天完成（支持流式和非流式）"""
            if not self.facade:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Service not initialized"
                )

            try:
                # 提取用户消息
                user_message = ""
                for msg in request.messages:
                    print("msg",msg)
                    if msg.role == "user":
                        user_message = msg.content


                if not user_message:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="No user message found"
                    )

                # 判断是否使用增强版认知流程
                use_enhanced = request.model == "zyantine-enhanced"

                # 调用自衍体系统
                result = self.facade.chat(
                    user_input=user_message,
                    use_enhanced_flow=use_enhanced,
                    stream=request.stream
                )

                if request.stream:
                    # 流式响应生成器
                    async def generate_stream():
                        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
                        created_time = int(time.time())
                        response_parts = []

                        # 发送第一个 chunk
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant"},
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {self._to_json(chunk)}\n\n"

                        # 分块发送内容
                        for part in result:
                            response_parts.append(part)
                            chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": part},
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {self._to_json(chunk)}\n\n"
                            await asyncio.sleep(0.01)

                        # 估算 token 数
                        response_text = "".join(response_parts)
                        prompt_tokens = sum(len(msg.content) // 4 for msg in request.messages)
                        completion_tokens = len(response_text) // 4
                        total_tokens = prompt_tokens + completion_tokens

                        # 发送完成 chunk
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }
                            ],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": total_tokens
                            }
                        }
                        yield f"data: {self._to_json(chunk)}\n\n"
                        yield "data: [DONE]\n\n"

                    return StreamingResponse(
                        generate_stream(),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                        }
                    )
                else:
                    # 非流式响应
                    response_text = result
                    
                    # 估算 token 数（简化版：1 token ≈ 4 字符）
                    prompt_tokens = sum(len(msg.content) // 4 for msg in request.messages)
                    completion_tokens = len(response_text) // 4
                    total_tokens = prompt_tokens + completion_tokens

                    # 构建响应
                    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
                    created_time = int(time.time())

                    return ChatCompletionResponse(
                        id=completion_id,
                        created=created_time,
                        model=request.model,
                        choices=[
                            ChatCompletionChoice(
                                index=0,
                                message=ChatMessage(
                                    role="assistant",
                                    content=response_text
                                ),
                                finish_reason="stop"
                            )
                        ],
                        usage=Usage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens
                        )
                    )

            except Exception as e:
                print(f"处理聊天完成时出错: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Internal server error: {str(e)}"
                )

        @self.app.post("/v1/chat/completions/stream")
        async def create_chat_completion_stream(request: ChatCompletionRequest):
            """创建聊天完成（流式）"""
            # 重定向到支持流式的主端点
            request.stream = True
            return await create_chat_completion(request)

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket, client_id: str = "default"):
            """WebSocket 端点 - 用于实时语音交互"""
            if not self.facade:
                await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="Service not initialized")
                return

            await self.connection_manager.connect(websocket, client_id)

            try:
                while True:
                    message = await websocket.receive()
                    
                    message_type = message.get("type")
                    
                    print(f"\n[DEBUG] 收到 WebSocket 消息 (client_id={client_id}):")
                    print(f"  消息类型: {message_type}")
                    
                    if message_type == "websocket.disconnect":
                        print(f"  客户端断开连接")
                        break
                    
                    if message_type == "websocket.receive":
                        text_data = message.get("text")
                        bytes_data = message.get("bytes")
                        
                        if text_data is not None:
                            data = text_data
                            print(f"  文本数据长度: {len(data)}")
                        elif bytes_data is not None:
                            data = bytes_data
                            print(f"  二进制数据长度: {len(data)}")
                            print(f"  二进制数据 (hex): {data.hex()}")
                            
                            try:
                                data = data.decode('utf-8')
                                print(f"  UTF-8 解码成功")
                            except UnicodeDecodeError as e:
                                print(f"  UTF-8 解码失败: {e}")
                                try:
                                    data = data.decode('gbk')
                                    print(f"  GBK 解码成功")
                                except UnicodeDecodeError as e2:
                                    print(f"  GBK 解码也失败: {e2}")
                                    try:
                                        await self.connection_manager.send_message(
                                            json.dumps({
                                                "type": "error",
                                                "message": "Failed to decode message: unsupported encoding"
                                            }, ensure_ascii=False),
                                            client_id
                                        )
                                    except Exception as e:
                                        print(f"发送错误消息失败: {e}")
                                        break
                                    continue
                        else:
                            print(f"  未知消息格式")
                            try:
                                await self.connection_manager.send_message(
                                    json.dumps({
                                        "type": "error",
                                        "message": "Unknown message format"
                                    }, ensure_ascii=False),
                                    client_id
                                )
                            except Exception as e:
                                print(f"发送错误消息失败: {e}")
                                break
                            continue
                    else:
                        print(f"  未知消息类型: {message_type}")
                        try:
                            await self.connection_manager.send_message(
                                json.dumps({
                                    "type": "error",
                                    "message": f"Unknown message type: {message_type}"
                                }, ensure_ascii=False),
                                client_id
                            )
                        except Exception as e:
                            print(f"发送错误消息失败: {e}")
                            break
                        continue
                    
                    print(f"  数据内容: {repr(data)}")
                    
                    try:
                        message_data = json.loads(data)
                        
                        print(f"\n[DEBUG] JSON 解析成功:")
                        print(f"  消息类型: {message_data.get('type')}")
                        print(f"  完整数据: {message_data}")
                        
                        if message_data.get("type") == "chat":
                            user_message = message_data.get("message", "")
                            model = message_data.get("model", "zyantine-v1")
                            
                            print(f"\n[DEBUG] 聊天消息处理:")
                            print(f"  用户消息: {repr(user_message)}")
                            print(f"  消息长度: {len(user_message)}")
                            print(f"  消息编码: {user_message.encode('utf-8')}")
                            print(f"  模型: {model}")
                            
                            if not user_message:
                                try:
                                    await self.connection_manager.send_message(
                                        json.dumps({
                                            "type": "error",
                                            "message": "Empty message"
                                        }, ensure_ascii=False),
                                        client_id
                                    )
                                except Exception as e:
                                    print(f"发送错误消息失败: {e}")
                                    break
                                continue

                            use_enhanced = model == "zyantine-enhanced"

                            print(f"\n[DEBUG] 调用 facade.chat()...")
                            response_text = self.facade.chat(
                                user_input=user_message,
                                use_enhanced_flow=use_enhanced
                            )
                            
                            print(f"\n[DEBUG] facade.chat() 返回:")
                            print(f"  响应: {repr(response_text)}")
                            print(f"  响应长度: {len(response_text)}")

                            response_json = json.dumps({
                                "type": "response",
                                "message": response_text,
                                "model": model,
                                "timestamp": int(time.time())
                            }, ensure_ascii=False)
                            
                            print(f"\n[DEBUG] 发送响应 JSON:")
                            print(f"  JSON: {repr(response_json)}")
                            
                            await self.connection_manager.send_message(response_json, client_id)

                        elif message_data.get("type") == "ping":
                            await self.connection_manager.send_message(
                                json.dumps({
                                    "type": "pong",
                                    "timestamp": int(time.time())
                                }, ensure_ascii=False),
                                client_id
                            )

                        elif message_data.get("type") == "close":
                            try:
                                await self.connection_manager.send_message(
                                    json.dumps({
                                        "type": "closing",
                                        "message": "Closing connection"
                                    }, ensure_ascii=False),
                                    client_id
                                )
                            except Exception as e:
                                print(f"发送关闭消息失败: {e}")
                            break

                        else:
                            try:
                                await self.connection_manager.send_message(
                                    json.dumps({
                                        "type": "error",
                                        "message": f"Unknown message type: {message_data.get('type')}"
                                    }, ensure_ascii=False),
                                    client_id
                                )
                            except Exception as e:
                                print(f"发送错误消息失败: {e}")
                                break

                    except json.JSONDecodeError as e:
                        print(f"\n[ERROR] JSON 解析失败:")
                        print(f"  错误: {e}")
                        print(f"  原始数据: {repr(data)}")
                        
                        try:
                            await self.connection_manager.send_message(
                                json.dumps({
                                    "type": "error",
                                    "message": f"Invalid JSON format: {str(e)}"
                                }, ensure_ascii=False),
                                client_id
                            )
                        except Exception as send_error:
                            print(f"发送错误消息失败: {send_error}")
                            break

                    except Exception as e:
                        print(f"\n[ERROR] 处理 WebSocket 消息时出错:")
                        print(f"  错误类型: {type(e).__name__}")
                        print(f"  错误信息: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        try:
                            await self.connection_manager.send_message(
                                json.dumps({
                                    "type": "error",
                                    "message": str(e)
                                }, ensure_ascii=False),
                                client_id
                            )
                        except Exception as send_error:
                            print(f"发送错误消息失败: {send_error}")
                            break

            except WebSocketDisconnect:
                self.connection_manager.disconnect(client_id)
                print(f"客户端 {client_id} 主动断开连接")

            except Exception as e:
                print(f"\n[ERROR] WebSocket 连接错误:")
                print(f"  错误类型: {type(e).__name__}")
                print(f"  错误信息: {e}")
                import traceback
                traceback.print_exc()
                self.connection_manager.disconnect(client_id)

    def _register_exception_handlers(self):
        """注册异常处理器"""

        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "message": exc.detail,
                        "type": "invalid_request_error",
                        "param": None,
                        "code": None
                    }
                }
            )

        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": {
                        "message": str(exc),
                        "type": "server_error",
                        "param": None,
                        "code": None
                    }
                }
            )

    def _to_json(self, obj: Any) -> str:
        """转换为 JSON 字符串（确保中文不转义）"""
        import json
        return json.dumps(obj, ensure_ascii=False)

    def run(self):
        """运行服务器"""
        app = self.create_app()
        print(f"\n{'='*60}")
        print(f"自衍体 AI API 服务")
        print(f"{'='*60}")
        print(f"服务地址: http://{self.host}:{self.port}")
        print(f"API 文档: http://{self.host}:{self.port}/docs")
        print(f"会话ID: {self.session_id}")
        print(f"{'='*60}\n")

        uvicorn.run(
            app,
            host=self.host,
            port=self.port,
            log_level="info"
        )


def main():
    """主函数"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="自衍体 AI API 服务")

    parser.add_argument("--host", "-H", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", "-p", type=int, default=8001, help="监听端口")
    parser.add_argument("--api-key", "-k", help="OpenAI API 密钥")
    parser.add_argument("--session", "-s", default="default", help="会话ID")
    parser.add_argument("--config", "-c", default="config/llm_config.json", help="配置文件路径")

    args = parser.parse_args()

    # 从配置文件读取API密钥和base_url
    api_key = None
    base_url = None
    config_path = args.config
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                
            # 获取当前启用的provider
            current_provider = config["api"]["provider"]
            if current_provider in config["api"]["providers"]:
                provider_config = config["api"]["providers"][current_provider]
                api_key = provider_config["api_key"]
                base_url = provider_config["base_url"]
                print(f"从配置文件 {config_path} 加载了 {current_provider} 服务的配置")
        except Exception as e:
            print(f"读取配置文件失败: {e}")
    else:
        print(f"配置文件 {config_path} 不存在")

    # 命令行参数优先
    if args.api_key:
        api_key = args.api_key

    server = APIServer(
        host=args.host,
        port=args.port,
        api_key=api_key,
        session_id=args.session,
        config_file=args.config
    )

    server.run()


if __name__ == "__main__":
    main()

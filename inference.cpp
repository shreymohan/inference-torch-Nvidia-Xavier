#include <torch/script.h> // One-stop header.
#include"cnpy.h"
#include <iostream>
#include <memory>

int main() {


  // Deserialize the ScriptModule from a file using torch::jit::load().
  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("/home/tecsar/torch_cpp/keytrack_trace.pt");
  //torch::load(model,"/home/tecsar/torch_cpp/keytrack_trace.pt");
  const cnpy::NpyArray data=cnpy::npy_load("./input.npy");
   float* datap=(float*)data.data<float>();
  //const_cast<float*>(reinterpret_cast<const float*>(datap));
  //(void*) datap=datap;
  at::Tensor tensor_data=torch::from_blob(datap, {3,32,54}, at::kFloat);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(tensor_data.to(at::kCUDA));
  at::Tensor output = module->forward(inputs).toTensor();
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/4) << '\n';
  //assert(module != nullptr);
  //std::cout << "ok\n";
}
